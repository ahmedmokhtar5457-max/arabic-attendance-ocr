# Arabic Attendance Data Extraction System - Deployment Guide

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4-core processor (Intel i5 or AMD Ryzen 5 equivalent)
- **RAM**: 8GB DDR4
- **Storage**: 50GB available disk space
- **Network**: Not required (fully offline system)

#### Recommended Requirements
- **CPU**: 8-core processor (Intel i7 or AMD Ryzen 7 equivalent)
- **RAM**: 16GB DDR4
- **Storage**: 100GB SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for faster processing)

#### Production Requirements (High Volume)
- **CPU**: 16-core processor (Intel Xeon or AMD EPYC)
- **RAM**: 32GB DDR4
- **Storage**: 500GB NVMe SSD
- **GPU**: NVIDIA RTX 3080 or better (8GB+ VRAM)

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04 LTS or later (recommended)
- **Windows**: Windows 10/11 Professional
- **macOS**: macOS 11.0 or later

#### Python Environment
- **Python**: 3.8 or later (3.9 recommended)
- **pip**: Latest version
- **Virtual environment**: venv or conda

## Installation Guide

### 1. System Preparation

#### Linux (Ubuntu/Debian)
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
sudo apt install -y libfontconfig1 libxrender1 libxtst6

# Install additional dependencies for PDF processing
sudo apt install -y poppler-utils

# For GPU support (optional)
# Follow NVIDIA CUDA installation guide for your system
```

#### Windows
```powershell
# Install Python 3.9 from python.org
# Install Microsoft Visual C++ Redistributable
# Install Git for Windows (optional)

# Install Windows Subsystem for Linux (WSL2) for better compatibility (optional)
wsl --install
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9
brew install poppler
```

### 2. Project Setup

```bash
# Clone or extract the project
cd /opt  # or your preferred directory
# Extract project files here

# Navigate to project directory
cd arabic_attendance_ocr

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Setup

```bash
# Create models directory
mkdir -p models

# Download pre-trained models (if available)
# Note: You may need to train the Arabic digit CNN model
python scripts/download_models.py

# Or train the digit recognition model
python scripts/train_digit_model.py
```

### 4. Configuration

```bash
# Create configuration file
cp config/config.example.json config/config.json

# Edit configuration as needed
nano config/config.json
```

Example configuration:
```json
{
    "use_gpu": false,
    "target_dpi": 300,
    "min_confidence": 0.5,
    "target_columns": [2, 3, 4],
    "max_images_per_pdf": 100,
    "validation_rules": {
        "attendance_time_range": [1, 24],
        "max_shift_id": 999,
        "require_all_fields": false
    },
    "logging": {
        "level": "INFO",
        "enable_file_logging": true,
        "log_dir": "logs"
    }
}
```

### 5. Service Setup

#### Option A: Direct Python Execution
```bash
# Start the service
python -m src.api.main
```

#### Option B: Using Gunicorn (Recommended for Production)
```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Option C: Using Docker
```bash
# Build Docker image
docker build -t arabic-attendance-ocr .

# Run container
docker run -d -p 8000:8000 -v $(pwd)/logs:/app/logs arabic-attendance-ocr
```

#### Option D: System Service (Linux)
Create systemd service file:
```bash
sudo nano /etc/systemd/system/arabic-attendance-ocr.service
```

Service file content:
```ini
[Unit]
Description=Arabic Attendance OCR Service
After=network.target

[Service]
Type=simple
User=ocr-user
WorkingDirectory=/opt/arabic_attendance_ocr
Environment=PATH=/opt/arabic_attendance_ocr/venv/bin
ExecStart=/opt/arabic_attendance_ocr/venv/bin/gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable arabic-attendance-ocr
sudo systemctl start arabic-attendance-ocr
```

## Performance Optimization

### 1. CPU Optimization
```bash
# Set CPU governor to performance mode (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase worker processes based on CPU cores
# In gunicorn: -w $(nproc)
```

### 2. Memory Optimization
```bash
# Increase swap if needed (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to /etc/fstab for persistence
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. GPU Optimization (if available)
```bash
# Install CUDA toolkit
# Follow NVIDIA's official installation guide

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Storage Optimization
```bash
# Use SSD for better I/O performance
# Mount with noatime option for better performance
# In /etc/fstab: /dev/sda1 /opt ext4 defaults,noatime 0 2

# Set up log rotation
sudo nano /etc/logrotate.d/arabic-attendance-ocr
```

Log rotation configuration:
```
/opt/arabic_attendance_ocr/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ocr-user ocr-user
    postrotate
        systemctl reload arabic-attendance-ocr
    endscript
}
```

## Security Configuration

### 1. User Setup
```bash
# Create dedicated user
sudo useradd -r -s /bin/false ocr-user
sudo chown -R ocr-user:ocr-user /opt/arabic_attendance_ocr
```

### 2. Firewall Configuration
```bash
# Allow only necessary ports (Linux with ufw)
sudo ufw enable
sudo ufw allow 8000/tcp  # API port
sudo ufw deny 22/tcp     # Disable SSH if not needed
```

### 3. File Permissions
```bash
# Set appropriate permissions
chmod 755 /opt/arabic_attendance_ocr
chmod 644 /opt/arabic_attendance_ocr/config/config.json
chmod 700 /opt/arabic_attendance_ocr/logs
```

## Monitoring and Maintenance

### 1. Health Monitoring
```bash
# Check service status
systemctl status arabic-attendance-ocr

# Monitor logs
tail -f /opt/arabic_attendance_ocr/logs/application.log

# Check API health
curl http://localhost:8000/health
```

### 2. Performance Monitoring
```bash
# Monitor system resources
htop
iotop
nvidia-smi  # For GPU monitoring

# Monitor API performance
curl http://localhost:8000/statistics
```

### 3. Backup Strategy
```bash
# Backup configuration and models
tar -czf backup_$(date +%Y%m%d).tar.gz config/ models/ logs/

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/arabic-ocr"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" /opt/arabic_attendance_ocr/config /opt/arabic_attendance_ocr/models
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### 2. Memory Issues
```bash
# Check memory usage
free -h

# Reduce worker processes
# In gunicorn: -w 2 instead of -w 4
```

#### 3. GPU Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Permission Issues
```bash
# Fix ownership
sudo chown -R ocr-user:ocr-user /opt/arabic_attendance_ocr

# Fix permissions
sudo chmod -R 755 /opt/arabic_attendance_ocr
```

### Log Analysis
```bash
# Check error logs
grep -i error /opt/arabic_attendance_ocr/logs/errors.log

# Check processing logs
tail -f /opt/arabic_attendance_ocr/logs/processing.log

# Check API access logs
grep "POST /process" /opt/arabic_attendance_ocr/logs/application.log
```

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use shared storage for models and configuration
- Implement job queue for processing distribution

### Vertical Scaling
- Increase CPU cores and RAM
- Add GPU acceleration
- Use faster storage (NVMe SSD)

### Performance Tuning
- Adjust worker processes based on CPU cores
- Optimize batch sizes for GPU processing
- Implement caching for frequently accessed data

## Maintenance Schedule

### Daily
- Check service status
- Monitor disk space
- Review error logs

### Weekly
- Analyze processing statistics
- Check system resource usage
- Update log rotation

### Monthly
- Update dependencies (security patches)
- Backup configuration and models
- Performance review and optimization

### Quarterly
- Full system backup
- Security audit
- Capacity planning review