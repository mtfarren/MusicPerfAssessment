import argparse
import librosa
import torch
import numpy as np
from torch.autograd import Variable
from models.PCConvNet import PCConvNet
from models.SpectralCRNN import SpectralCRNN_Reg_big

# Get the path to the saved model and audio data from user input at the command line
# User will also specify which model architecture to use (0 for pitch contour model, 1 for mel spectrogram model)
parser = argparse.ArgumentParser(description = 'Enter the paths to a saved model and an audio file for which you want to generate a prediction. Also specify the model type to use (see below).')
parser.add_argument('modeldir', metavar = 'model_dir', type = str, nargs = 1, help='Path to saved model')
parser.add_argument('audiodir', metavar = 'audio_dir', type = str, nargs = 1, help='Path to audio data')
parser.add_argument('model_type', metavar = 'model_type', type = int, nargs = 1, help='Model for prediction. 0: PC-FCN, 1: M-CRNN')
args = parser.parse_args()
model_dir = args.modeldir[0]
audio_dir = args.audiodir[0]
model_type = args.model_type[0]

def extract_mel():
    """
    Extract the mel spectrogram for the audio.
    """
    y, sr = librosa.load(audio_dir)

    # Normalize data in the same way as when we trained
    rms = np.sqrt(np.mean(y * y))
    if rms > 1e-4:
        y = y/rms

    x = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=1024, n_mels=96)
    x = librosa.amplitude_to_db(x)
        # Equivalent to librosa.logamplitude(x ** 2)

    return torch.FloatTensor(x)

def extract_pitch():
    """
    Extract the pitch contour for the audio.
    Make sure this function returns pitch contour as a tensor.
    """
    pass

def compute_score():
    """
    Run the specified neural network on the provided audio sample.
    This function will return a normalized assessment score (float between 0 and 1) for the provided audio.
    """
    # For the pitch contour model
    if model_type == 0:
        pc_model = PCConvNet(0)
        if torch.cuda.is_available():
            pc_model.cuda()
            pc_model.load_state_dict(torch.load(model_dir))
        else:
            pc_model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
        pc_model.eval()

        model_input = extract_pitch()
        # Convert to cuda tensor if cuda available
        if torch.cuda.is_available():
            model_input = model_input.cuda()
        # Wrap tensor in pytorch Variable
        model_input = Variable(model_input)

        # Compute output
        model_output = pc_model(model_input)
        output = model_output.tolist()[0][0]

        return output

    # For the mel spectrogram model
    elif model_type == 1:
        mel_model = SpectralCRNN_Reg_big()
        if torch.cuda.is_available():
            mel_model.cuda()
            mel_model.load_state_dict(torch.load(model_dir).state_dict())
        else:
            mel_model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage).state_dict())
        mel_model.eval()

        model_input = extract_mel()
        # Convert the model input to a 4D tensor
        model_input = model_input.unsqueeze(0)
        model_input = model_input.unsqueeze(0)

        # Convert to cuda tensor if cuda available
        if torch.cuda.is_available():
            model_input = model_input.cuda()
        # Wrap tensor in pytorch Variable
        model_input = Variable(model_input)

        mel_model.init_hidden(model_input.size(0))

        # Compute output
        model_output = mel_model(model_input)
        output = model_output.data.tolist()[0][0]

        return output

#print(compute_score())


