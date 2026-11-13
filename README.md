# English-Urdu Machine Translation

A comparative study of sequence-to-sequence neural translation models for English-to-Urdu machine translation.

This project explores and compares multiple architectures for translating English sentences into Urdu, including:

- RNN
- BiRNN
- LSTM
- BiLSTM
- Transformer

The work is implemented in a Jupyter notebook and includes preprocessing, tokenization, training, and evaluation using BLEU and ROUGE-L metrics.

## Overview

The repository implements an English-to-Urdu neural machine translation pipeline using encoder-decoder architectures. The study focuses on how different recurrent and attention-based models affect translation quality on a parallel English-Urdu corpus.

The notebook performs the following steps:

- Load the parallel English-Urdu dataset
- Clean and normalize sentence pairs
- Tokenize both languages
- Build vocabulary and sequence padding
- Train multiple model variants
- Save model checkpoints and tokenizers
- Evaluate translations with BLEU and ROUGE-L

## Project Structure

```text
.
├── NLP_Assignment_3.ipynb        # Main experiment notebook
├── README.md                    # Project documentation
├── config.json                  # Model/training configuration
├── part3_translation_evaluation.csv
│                                # Evaluation summary for each model
├── models/
│   ├── rnn_seq2seq.keras
│   ├── birnn_seq2seq.keras
│   ├── lstm_seq2seq.keras
│   ├── bilstm_seq2seq.keras
│   └── transformer_seq2seq.keras
├── tokenizers/
│   ├── english_tokenizer.pkl
│   └── urdu_tokenizer.pkl
└── .gitignore
```

## Dataset

The project uses a parallel English-Urdu corpus stored in the dataset folder referenced from the notebook.

The preprocessing pipeline:

- reads the English and Urdu text files
- removes empty or mismatched rows
- strips unwanted whitespace
- builds sentence pair DataFrames
- prepares sequences for model training

## Model Configuration

The model settings used in the notebook are stored in `config.json`:

```json
{
  "MAX_ENGLISH_LENGTH": 14,
  "MAX_URDU_LENGTH": 21,
  "english_vocab_size": 5680,
  "urdu_vocab_size": 5935,
  "embedding_dim": 128,
  "hidden_units": 256,
  "epochs": 50,
  "batch_size": 64
}
```

These settings define the sequence length limits, vocabulary size, embedding dimension, hidden units, and training hyperparameters.

## Training Pipeline

The notebook builds several sequence-to-sequence models using TensorFlow/Keras:

- Encoder-decoder framework
- Embedding layers
- Recurrent layers for sequence modeling
- Decoder output layer with softmax activation
- Training with sparse categorical cross-entropy

The project trains and compares the following architectures:

1. RNN Seq2Seq
2. BiRNN Seq2Seq
3. LSTM Seq2Seq
4. BiLSTM Seq2Seq
5. Transformer Seq2Seq

## Evaluation Metrics

Model performance is evaluated using:

- BLEU score
- ROUGE-L score

## Results

The saved evaluation results in `part3_translation_evaluation.csv` show the following:

- RNN: BLEU = 0.2471, ROUGE-L = 0.5184
- BiRNN: BLEU = 0.2265, ROUGE-L = 0.5197
- LSTM: BLEU = 0.2792, ROUGE-L = 0.5857
- BiLSTM: BLEU = 0.3130, ROUGE-L = 0.6182
- Transformer: BLEU = 0.3858, ROUGE-L = 0.6670

The Transformer model achieves the best overall performance in this comparison.

## Dependencies

This project depends on the following Python libraries:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

Additional notebook support may require:

```bash
pip install jupyter
```

## Usage

1. Clone the repository.
2. Open `NLP_Assignment_3.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells in order.
4. Trained models and tokenizers will be generated or loaded from the repository folders.

To reproduce the experiments:

```bash
git clone https://github.com/AbdulGhani9087/English-Urdu-Machine-Translation.git
cd English-Urdu-Machine-Translation
jupyter notebook NLP_Assignment_3.ipynb
```

## Notes

- The project is focused on machine translation quality comparison rather than deployment.
- Saved model artifacts are available in `models/` for direct reuse.
- Tokenizers are stored in `tokenizers/` so text preprocessing remains consistent during inference.

## License

This repository does not currently include a license file. If you plan to reuse or distribute the project, consider adding an appropriate license.

## Author

Abdul Ghani

## Repository

https://github.com/AbdulGhani9087/English-Urdu-Machine-Translation
