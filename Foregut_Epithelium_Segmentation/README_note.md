# Video Segmentation with Segmentation Anything 2

This code is for video frame segmentation based on [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2).  
It allows you to extract frames from a video, apply segmentation, and reconstruct a processed video.

## Requirements

- **Python** ≥ 3.10  
- **PyTorch** ≥ 2.5.1  
- **Torchvision** ≥ 0.20.1  
- [Segmentation Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) must be installed before running this code.  


## Installation

Make sure SAM2 is properly installed following the instructions in its repository.

---

## Usage

### Step 1: Extract Video Frames

Inside the `example_case/frame/` folder, extract video frames using **ffmpeg**:

```bash
ffmpeg -i ../example_case.avi -q:v 2 -start_number 0 '%05d.jpg'
```

This will create sequentially numbered images (e.g., `00000.jpg`, `00001.jpg`, …) from the video.

---

### Step 2: Run Segmentation

Open the Jupyter Notebook (video_segmentation.ipynb) and run each cell.  
This will generate processed segmented images and reassemble them into a video.

---

## Notes

- For best performance, ensure you have a CUDA-enabled GPU with compatible PyTorch installed.  
- Modify `ffmpeg` and notebook parameters as needed for your dataset.  

---

## Contact

If you have any questions, please feel free to reach out:  
- **Name**: Deng Li  
- **Email**: li.den@northeastern.edu
