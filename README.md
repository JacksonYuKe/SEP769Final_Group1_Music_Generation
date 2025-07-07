# Music Generation using Deep Learning
**SEP769 Final Project - Group 1**

This repository contains three different deep learning approaches for generating piano music from MIDI files. Each approach explores a different neural network architecture to learn musical patterns and generate new compositions.

## 🎵 Overview

The project implements three distinct models for music generation:

1. **LSTM Model** (`Final_769Lstm.ipynb`) - Sequential music generation using Long Short-Term Memory networks
2. **WaveNet Model** (`Final_769WaveNet.ipynb`) - Convolutional approach with dilated convolutions for music generation
3. **Transformer Model** (`Transformer_Music_Generator_final.ipynb`) - Self-attention based model for piano roll generation

## 📁 Project Structure

```
SEP769Final_Group1_Music_Generation/
├── Final_769Lstm.ipynb              # LSTM-based music generation
├── Final_769WaveNet.ipynb           # WaveNet-based music generation  
├── Transformer_Music_Generator_final.ipynb  # Transformer-based generation
├── README.md                        # Project documentation
└── .claude/                         # Claude CLI configuration
    └── settings.local.json
```

## 🚀 Models

### 1. LSTM Model (`Final_769Lstm.ipynb`)
- **Architecture**: Single LSTM layer (128 units) with three output heads
- **Dataset**: Maestro v2.0.0 dataset (classical piano performances)
- **Input**: Sequences of pitch, step, and duration values
- **Output**: Generates next note predictions for pitch (categorical), step and duration (regression)
- **Features**:
  - Custom loss function with positive pressure for step/duration
  - Temperature-controlled sampling for pitch generation
  - Weighted loss balancing between pitch, step, and duration

### 2. WaveNet Model (`Final_769WaveNet.ipynb`)
- **Architecture**: Convolutional neural network with dilated convolutions
- **Dataset**: Custom collection of popular piano MIDI files
- **Input**: Sequences of note representations (32 timesteps)
- **Output**: Next note prediction using categorical classification
- **Features**:
  - Embedding layer for note representation
  - Multiple Conv1D layers with increasing dilation rates (1, 2, 4)
  - Dropout regularization and max pooling
  - GlobalMaxPooling for feature aggregation

### 3. Transformer Model (`Transformer_Music_Generator_final.ipynb`)
- **Architecture**: Custom transformer with multiheaded self-attention
- **Dataset**: Custom MIDI collection converted to piano roll format
- **Input**: Piano roll sequences (88 piano keys × timesteps)
- **Output**: Generated piano roll representations
- **Features**:
  - Custom multiheaded attention with relative positional embeddings
  - Piano roll representation (88 keys)
  - Top-p (nucleus) sampling for diverse generation
  - Batch normalization between attention layers

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Core dependencies
pip install tensorflow torch torchvision
pip install pretty_midi mido music21
pip install numpy pandas matplotlib seaborn scikit-learn
pip install pyfluidsynth

# System dependencies (Ubuntu/Debian)
sudo apt install fluidsynth timidity ffmpeg

# For Jupyter notebooks
pip install jupyter ipython
```

### Google Colab Setup
The notebooks are designed to run in Google Colab with GPU acceleration:

1. Upload notebooks to Google Colab
2. Mount Google Drive for dataset access
3. Install dependencies using `!pip install` commands in the notebooks
4. Run cells sequentially

## 📊 Datasets

### LSTM Model
- **Maestro v2.0.0**: 1,282 classical piano MIDI files
- **Processing**: Extracts pitch, step, duration sequences
- **Download**: Automatically downloaded via TensorFlow

### WaveNet Model  
- **Custom Collection**: 21 popular piano songs (various artists)
- **Processing**: Converts to note sequences, filters frequent notes (≥50 occurrences)
- **Features**: 56 unique frequent notes from 215 total unique notes

### Transformer Model
- **Custom MIDI Collection**: Piano pieces in various styles
- **Processing**: Converts to piano roll format (20 fps, 88 keys)
- **Features**: 533 unique piano roll "words" (note combinations)

## 🎼 Usage

### Running the Models

1. **LSTM Model**:
   ```python
   # Set parameters
   temperature = 5.0  # Controls randomness (higher = more random)
   num_predictions = 240  # Number of notes to generate
   
   # Generate music
   pitch, step, duration = predict_next_note(input_notes, model, temperature)
   ```

2. **WaveNet Model**:
   ```python
   # Generate 20 new notes
   predictions = []
   for i in range(20):
       prob = model.predict(random_music)[0]
       y_pred = np.argmax(prob, axis=0)
       predictions.append(y_pred)
   ```

3. **Transformer Model**:
   ```python
   # Set generation parameters
   temp = 1.5  # Temperature for sampling
   top_p = 0.99  # Top-p sampling threshold
   
   # Generate piano roll
   gen = generate_music(model, length, temp, top_p)
   ```

### Output Formats

All models generate MIDI files that can be:
- Played directly in notebooks using `IPython.display.Audio`
- Downloaded as `.mid` files
- Converted to audio formats using FluidSynth/Timidity

## 📈 Model Performance

### LSTM Model
- **Training**: 50 epochs with early stopping
- **Loss Weighting**: Pitch (0.05), Step (1.0), Duration (1.0)
- **Architecture**: 84,354 trainable parameters

### WaveNet Model
- **Training**: 40 epochs with model checkpointing
- **Architecture**: 228,312 trainable parameters
- **Validation**: 80/20 train-validation split

### Transformer Model
- **Training**: 6 epochs with batch size 256
- **Architecture**: Custom multiheaded attention with relative positioning
- **Performance**: Tracks both accuracy and loss per batch/epoch

## 🎯 Key Features

- **Multiple Architectures**: Compare RNN, CNN, and Transformer approaches
- **Different Representations**: Note sequences vs. piano roll formats
- **Flexible Generation**: Temperature and top-p sampling controls
- **Audio Output**: Direct MIDI playback and download capabilities
- **Visualization**: Piano roll plots and training metrics

## 🔧 Customization

### Adjusting Generation Parameters

- **Temperature**: Higher values (>1) increase randomness, lower values (<1) make output more deterministic
- **Sequence Length**: Modify `seq_length` for different context windows
- **Top-p Sampling**: Adjust nucleus sampling threshold for diversity control

### Model Architecture Changes

- **LSTM**: Modify layer sizes, add more LSTM layers, adjust loss weights
- **WaveNet**: Change dilation rates, add more convolutional layers, modify embedding dimensions
- **Transformer**: Adjust number of heads, layers, or embedding dimensions

## 📝 Notes

- Models are designed for piano music generation
- MIDI files should contain piano tracks for best results
- GPU acceleration recommended for training (especially Transformer model)
- Generated music quality depends on training data and model parameters

## 🤝 Contributing

This is an academic project for SEP769. The implementations serve as educational examples of different approaches to music generation using deep learning.

## 📄 License

Academic use only - SEP769 Final Project, Group 1