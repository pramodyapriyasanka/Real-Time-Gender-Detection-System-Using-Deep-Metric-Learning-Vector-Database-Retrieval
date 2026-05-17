import sys
import pickle
from pathlib import Path

import cv2
import numpy as np
import faiss
import onnxruntime as ort
import yaml

try:
    from ultralytics import YOLO
    import torch
    import torch.nn as nn
except Exception:
    YOLO = None
    torch = None
    nn = None


class ArcFaceONNX:
    def __init__(self, model_path: str):
        """Initialize ArcFace model with GPU acceleration"""
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider',
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (112, 112)
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess face crop for ArcFace"""
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        return img
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract embedding from single face"""
        preprocessed = self.preprocess(img).reshape(1, 3, 112, 112)
        embedding = self.session.run([self.output_name], {self.input_name: preprocessed})[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()


class MLPGenderClassifier(nn.Module):
    """MLP Neural Network for Gender Classification from 512D embeddings"""
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], dropout=0.3):
        super(MLPGenderClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: Female, Male
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RealtimeGenderDetector:
    def __init__(self, yolo_model_path: str, arcface_model_path: str, 
                 faiss_index_path: str, metadata_path: str, mlp_model_path: str = None,
                 use_ensemble: bool = True, config: dict = None):
        
        # Store config
        self.config = config or {}
        
        # Get thresholds from config
        self.face_conf_threshold = self.config.get('face_detection', {}).get('confidence_threshold', 0.6)
        self.male_threshold = self.config.get('gender_classification', {}).get('male_threshold', 0.5)
        self.female_threshold = self.config.get('gender_classification', {}).get('female_threshold', 0.5)
        self.k_neighbors = self.config.get('gender_classification', {}).get('k_neighbors', 5)
        self.faiss_weight = self.config.get('gender_classification', {}).get('faiss_weight', 0.5)
        self.mlp_weight = self.config.get('gender_classification', {}).get('mlp_weight', 0.5)
        
        # Check GPU
        self.device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        if self.device == 'cpu':
            print("WARNING: GPU not available, using CPU (will be slower)")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        print("Loading YOLO face detection model...")
        self.face_detector = YOLO(yolo_model_path)
        
        print("Loading ArcFace embedding model...")
        self.arcface = ArcFaceONNX(arcface_model_path)
        
        print("Loading FAISS index...")
        self.index = faiss.read_index(faiss_index_path)
        
        print("Loading metadata...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded {self.index.ntotal} embeddings from database")
        
        # Load MLP model if provided
        self.mlp_model = None
        self.use_ensemble = use_ensemble and mlp_model_path is not None
        
        if mlp_model_path and Path(mlp_model_path).exists():
            print("Loading MLP gender classifier...")
            self.mlp_model = MLPGenderClassifier()
            checkpoint = torch.load(mlp_model_path, map_location=self.device)
            self.mlp_model.load_state_dict(checkpoint['model_state_dict'])
            self.mlp_model.to(self.device)
            self.mlp_model.eval()
            print(f"MLP model loaded (Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")
            if self.use_ensemble:
                print("Ensemble mode: ENABLED (combining FAISS + MLP)")
        else:
            print("MLP model not found, using FAISS-only mode")
            self.use_ensemble = False
        
        print("Ready for real-time detection!\n")
    
    def predict_gender_faiss(self, embedding: np.ndarray, k: int = 5) -> tuple:
        """Predict gender using FAISS k-NN"""
        try:
            # Search FAISS for k nearest neighbors
            distances, indices = self.index.search(embedding.reshape(1, -1), k)
            
            # Count votes from neighbors
            gender_votes = {'Female': 0, 'Male': 0}
            for idx in indices[0]:
                if idx < len(self.metadata):
                    gender = self.metadata[idx]['class']
                    gender_votes[gender] += 1
            
            # Calculate probability
            female_prob = gender_votes['Female'] / k
            male_prob = gender_votes['Male'] / k
            
            return female_prob, male_prob
        
        except Exception as e:
            print(f"Error in FAISS prediction: {e}")
            return 0.5, 0.5
    
    def predict_gender_mlp(self, embedding: np.ndarray) -> tuple:
        """Predict gender using MLP model"""
        try:
            with torch.no_grad():
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                output = self.mlp_model(embedding_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                # probs[0] = Female, probs[1] = Male
                return float(probs[0]), float(probs[1])
        
        except Exception as e:
            print(f"Error in MLP prediction: {e}")
            return 0.5, 0.5
    
    def predict_gender(self, face_crop: np.ndarray, k: int = 5, 
                      faiss_weight: float = 0.5, mlp_weight: float = 0.5) -> tuple:
        """Predict gender using ensemble of FAISS and MLP (if available)"""
        try:
            # Extract embedding
            embedding = self.arcface.extract(face_crop)
            
            if self.use_ensemble and self.mlp_model is not None:
                # Get predictions from both models
                faiss_female_prob, faiss_male_prob = self.predict_gender_faiss(embedding, k)
                mlp_female_prob, mlp_male_prob = self.predict_gender_mlp(embedding)
                
                # Ensemble: weighted average
                female_prob = faiss_weight * faiss_female_prob + mlp_weight * mlp_female_prob
                male_prob = faiss_weight * faiss_male_prob + mlp_weight * mlp_male_prob
                
                # Determine final prediction with thresholds
                if female_prob >= self.female_threshold and female_prob > male_prob:
                    predicted_gender = "Female"
                    confidence = female_prob
                elif male_prob >= self.male_threshold and male_prob > female_prob:
                    predicted_gender = "Male"
                    confidence = male_prob
                else:
                    predicted_gender = "Unknown"
                    confidence = max(female_prob, male_prob)
                
                # Return with detailed info
                return predicted_gender, confidence, {
                    'faiss': (faiss_female_prob, faiss_male_prob),
                    'mlp': (mlp_female_prob, mlp_male_prob),
                    'ensemble': (female_prob, male_prob)
                }
            else:
                # FAISS-only mode
                female_prob, male_prob = self.predict_gender_faiss(embedding, k)
                
                if female_prob >= self.female_threshold and female_prob > male_prob:
                    predicted_gender = "Female"
                    confidence = female_prob
                elif male_prob >= self.male_threshold and male_prob > female_prob:
                    predicted_gender = "Male"
                    confidence = male_prob
                else:
                    predicted_gender = "Unknown"
                    confidence = max(female_prob, male_prob)
                
                return predicted_gender, confidence, None
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Unknown", 0.0, None
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = None) -> np.ndarray:
        """Process single frame with face detection and gender classification"""
        # Use config threshold if not provided
        if conf_threshold is None:
            conf_threshold = self.face_conf_threshold
        
        # Detect faces
        results = self.face_detector(frame, conf=conf_threshold, verbose=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Process each detected face
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box
                xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf.cpu().numpy())
                
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    # Crop face
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Predict gender
                    result = self.predict_gender(face_crop, k=self.k_neighbors, 
                                                 faiss_weight=self.faiss_weight, 
                                                 mlp_weight=self.mlp_weight)
                    gender, gender_conf = result[0], result[1]
                    details = result[2] if len(result) > 2 else None
                    
                    # Draw bounding box with color based on gender
                    if gender == "Female":
                        color = tuple(self.config.get('colors', {}).get('female', [255, 0, 255]))
                    elif gender == "Male":
                        color = tuple(self.config.get('colors', {}).get('male', [255, 255, 0]))
                    else:
                        color = tuple(self.config.get('colors', {}).get('unknown', [128, 128, 128]))
                    
                    box_thickness = self.config.get('display', {}).get('box_thickness', 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    
                    # Prepare label
                    show_confidence = self.config.get('display', {}).get('show_confidence', True)
                    show_detailed = self.config.get('display', {}).get('show_detailed_info', True)
                    font_scale = self.config.get('display', {}).get('font_scale', 0.6)
                    
                    if show_confidence:
                        label = f"{gender} {gender_conf*100:.0f}%"
                    else:
                        label = gender
                    
                    if self.use_ensemble and details and show_detailed:
                        # Add smaller text with model details
                        faiss_f, faiss_m = details['faiss']
                        mlp_f, mlp_m = details['mlp']
                        detail_text = f"F:{faiss_f*100:.0f}/{mlp_f*100:.0f} M:{faiss_m*100:.0f}/{mlp_m*100:.0f}"
                    else:
                        detail_text = None
                    
                    # Draw label background
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                    )
                    
                    bg_height = label_height + 10
                    if detail_text:
                        bg_height += 20
                    
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - bg_height), 
                        (x1 + label_width + 10, y1),
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1 + 5, y1 - 5 - (20 if detail_text else 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        (0, 0, 0), 
                        2
                    )
                    
                    # Draw detail text if ensemble
                    if detail_text:
                        cv2.putText(
                            frame, 
                            detail_text, 
                            (x1 + 5, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale * 0.67, 
                            (0, 0, 0), 
                            1
                        )
        
        return frame
    
    def run_webcam(self, camera_id: int = None):
        """Run real-time detection on webcam feed"""
        # Use config camera_id if not provided
        if camera_id is None:
            camera_id = self.config.get('video', {}).get('camera_id', 0)
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        # Set resolution from config
        width = self.config.get('video', {}).get('resolution_width', 1280)
        height = self.config.get('video', {}).get('resolution_height', 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("Starting webcam... Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Add FPS counter
            frame_count += 1
            cv2.putText(
                frame, 
                f"Frame: {frame_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display instructions
            cv2.putText(
                frame, 
                "Press 'q' to quit", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Show frame
            cv2.imshow('Real-time Gender Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed")
    
    def process_video(self, video_path: str, output_path: str = None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Write or display
            if writer:
                writer.write(frame)
            else:
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
            print(f"\nVideo saved to: {output_path}")
        cv2.destroyAllWindows()
        print("\nVideo processing complete")


def main():
    # Paths
    base_dir = Path(r"C:\Users\menuk\Desktop\AGE")
    config_path = base_dir / "config.yaml"
    
    # Load configuration
    print("Loading configuration...")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
    else:
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        config = {}
    
    # Get paths from config or use defaults
    yolo_model = base_dir / config.get('models', {}).get('yolo_model', 'Pre-Train Models/yolov11l-face.pt')
    arcface_model = base_dir / config.get('models', {}).get('arcface_model', 'Pre-Train Models/models/buffalo_l/w600k_r50.onnx')
    faiss_index = base_dir / config.get('models', {}).get('faiss_index', 'embeddings/face_embeddings.index')
    metadata = base_dir / config.get('models', {}).get('metadata', 'embeddings/metadata.pkl')
    mlp_model = base_dir / config.get('models', {}).get('mlp_model', 'models/mlp_gender_model.pth')
    
    # Check required files exist
    for path, name in [(yolo_model, "YOLO model"), (arcface_model, "ArcFace model"), 
                       (faiss_index, "FAISS index"), (metadata, "Metadata")]:
        if not path.exists():
            print(f"Error: {name} not found at {path}")
            sys.exit(1)
    
    # Check if MLP model exists (optional)
    if not mlp_model.exists():
        print(f"Warning: MLP model not found at {mlp_model}")
        print("Run 'python train_mlp_model.py' to train the MLP model")
        print("Continuing with FAISS-only mode...\n")
        mlp_model = None
    
    # Initialize detector
    use_ensemble = config.get('gender_classification', {}).get('use_ensemble', True)
    detector = RealtimeGenderDetector(
        str(yolo_model),
        str(arcface_model),
        str(faiss_index),
        str(metadata),
        str(mlp_model) if mlp_model else None,
        use_ensemble=use_ensemble,
        config=config
    )
    
    # Display configuration
    print("\n" + "="*60)
    print("Configuration Settings:")
    print("="*60)
    print(f"Face Detection Confidence: {detector.face_conf_threshold}")
    print(f"Male Threshold: {detector.male_threshold}")
    print(f"Female Threshold: {detector.female_threshold}")
    print(f"k-Neighbors: {detector.k_neighbors}")
    if detector.use_ensemble:
        print(f"Ensemble Weights: FAISS={detector.faiss_weight}, MLP={detector.mlp_weight}")
    print("="*60 + "\n")
    
    # Run webcam
    print("Starting real-time gender detection...")
    
    try:
        detector.run_webcam(camera_id=0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
