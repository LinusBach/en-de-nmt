from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import *
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
