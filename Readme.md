# Project Name: Seq2Seq Model for Machine Translation

## Introduction

This project contains a Seq2Seq model with attention for machine translation. The project includes several Python scripts for data preprocessing, model training, and evaluation.

## Dependencies

This project uses the following libraries:

- `PyTorch` (for the model and overall framework)
- `transformers` (for tokenization)
- `numpy` (for mathematical operations)
- `nltk` (for BLEU score computation)
- `matplotlib` (for plotting)
- `tqdm` (for progress bars)

Please ensure you have all these libraries installed. If not, you can install them using pip:

```bash
pip install torch transformers numpy nltk matplotlib tqdm
```

## Data

You will need two text files containing paired sentences in English (`train.en`) and German (`train.de`) for training. The files can be downloaded from the [Stanford Neural Machine Translation](https://nlp.stanford.edu/projects/nmt/) website and should be placed in a directory named "data" in the project root. 

## Usage

1. Preprocess your data using the `dataloader.py` script. This script will tokenize your data and save it in a suitable format for training. 

2. Train your model using the `train.py` script. This will save model parameters after each epoch in the "models" directory. The script will also plot the training loss after each epoch and save the plot as a .png file in the "plots" directory. 

3. Evaluate your model using the `evaluate_models.py` script. This script loads saved model parameters from the "models" directory and computes the loss on a validation set. 

## Known Issues and Limitations

- Data shuffling: The current implementation does not shuffle the data before training. This means that if there is very similar data over continuous stretches in the data file, the model may overfit to these sequences.

- No batching: Currently, we train on one example at a time, which is inefficient and slows down the training process. Implementing batching would likely improve the speed of training.

- Training steps: Due to the limitations mentioned above, we were only able to train each model for between 40k to 200k steps. 

## Future Work

Future improvements could include implementing data shuffling and batch training. Furthermore, experimenting with different architectures, such as transformer models, could potentially improve performance.

## Additional Resources

- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers library](https://huggingface.co/docs/transformers/index)
- [Seq2Seq models with attention tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [BLEU score computation](https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.corpus_bleu)
