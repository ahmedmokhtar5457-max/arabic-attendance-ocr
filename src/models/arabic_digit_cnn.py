"""
Arabic Handwritten Digit Recognition CNN Model

Custom CNN model optimized for Arabic handwritten digits (0-9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ArabicDigitCNN(nn.Module):
    """CNN model for Arabic handwritten digit recognition."""
    
    def __init__(self, num_classes: int = 10, input_size: Tuple[int, int] = (32, 32)):
        """
        Initialize Arabic digit CNN.
        
        Args:
            num_classes: Number of digit classes (0-9)
            input_size: Input image size (height, width)
        """
        super(ArabicDigitCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self) -> int:
        """Calculate the size after convolution and pooling layers."""
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.input_size)
            x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
            x = self.pool(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
            x = self.pool(F.relu(self.bn5(self.conv5(x))))
            return x.numel()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ArabicDigitDataset(Dataset):
    """Dataset class for Arabic handwritten digits."""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            images: List of digit images
            labels: List of corresponding labels (0-9)
            transform: Optional transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ArabicDigitRecognizer:
    """Arabic digit recognition system."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize digit recognizer.
        
        Args:
            model_path: Path to trained model file
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = ArabicDigitCNN().to(self.device)
        self.model_path = model_path
        
        # Arabic to English digit mapping
        self.arabic_to_english = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for digit recognition.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply transform
        tensor = self.transform(cleaned)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def recognize_digit(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize single digit in image.
        
        Args:
            image: Input image containing digit
            
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Get prediction
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_digit = str(predicted.item())
            confidence_score = confidence.item()
            
            return predicted_digit, confidence_score
    
    def recognize_number_sequence(self, images: List[np.ndarray]) -> Tuple[str, float]:
        """
        Recognize sequence of digits to form a number.
        
        Args:
            images: List of digit images in order
            
        Returns:
            Tuple of (recognized_number, average_confidence)
        """
        digits = []
        confidences = []
        
        for image in images:
            digit, confidence = self.recognize_digit(image)
            digits.append(digit)
            confidences.append(confidence)
        
        # Combine digits into number
        number = ''.join(digits)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return number, avg_confidence
    
    def convert_arabic_to_english_digits(self, text: str) -> str:
        """
        Convert Arabic digits to English digits.
        
        Args:
            text: Text containing Arabic digits
            
        Returns:
            Text with English digits
        """
        result = text
        for arabic, english in self.arabic_to_english.items():
            result = result.replace(arabic, english)
        return result
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset,
                   epochs: int = 50, batch_size: int = 32, 
                   learning_rate: float = 0.001) -> Dict:
        """
        Train the digit recognition model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.model_path:
                    self.save_model(self.model_path)
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}] - '
                           f'Train Loss: {train_loss/len(train_loader):.4f}, '
                           f'Train Acc: {train_acc:.2f}%, '
                           f'Val Loss: {val_loss/len(val_loader):.4f}, '
                           f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return history
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'num_classes': self.model.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


class DigitSegmenter:
    """Segments multi-digit numbers into individual digits."""
    
    def __init__(self):
        """Initialize digit segmenter."""
        pass
    
    def segment_digits(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment multi-digit number into individual digits.
        
        Args:
            image: Image containing number
            
        Returns:
            List of individual digit images
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (digits should be white on black)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            
            # Filter based on size and aspect ratio
            if area > 50 and 0.2 < aspect_ratio < 2.0:
                digit_contours.append((x, y, w, h))
        
        # Sort contours from left to right (for Arabic: right to left)
        digit_contours.sort(key=lambda c: c[0])
        
        # Extract digit images
        digit_images = []
        for x, y, w, h in digit_contours:
            # Add padding
            padding = 2
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            digit_img = gray[y1:y2, x1:x2]
            digit_images.append(digit_img)
        
        return digit_images