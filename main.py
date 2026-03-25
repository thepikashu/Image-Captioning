# Main execution functions
def main_training_pipeline():
    """Main function to train both models"""
    print("Starting Image Captioning Training Pipeline")
    print("=" * 50)

    # Train LSTM model
    print("Phase 1: Training LSTM Model")
    lstm_checkpoint = train_lstm_model(ModelConfiguration.TRAINING_EPOCHS)
    print(f"LSTM model saved to: {lstm_checkpoint}")

    # Train Transformer model
    print("\nPhase 2: Training Transformer Model")
    transformer_checkpoint = train_transformer_model(ModelConfiguration.TRAINING_EPOCHS)
    print(f"Transformer model saved to: {transformer_checkpoint}")

    return lstm_checkpoint, transformer_checkpoint


def main_evaluation_pipeline(lstm_checkpoint: str, transformer_checkpoint: str):
    """Main function to evaluate both models"""
    print("Starting Model Evaluation Pipeline")
    print("=" * 50)

    # Evaluate LSTM model
    print("Evaluating LSTM Model:")
    lstm_metrics = run_comprehensive_evaluation(lstm_checkpoint, "lstm")

    # Generate sample captions for LSTM
    print("\nSample LSTM Captions:")
    lstm_samples = generate_sample_captions(lstm_checkpoint, "lstm", 5, "val")

    print("\n" + "=" * 50)

    # Evaluate Transformer model
    print("Evaluating Transformer Model:")
    transformer_metrics = run_comprehensive_evaluation(transformer_checkpoint, "transformer")

    # Generate sample captions for Transformer
    print("\nSample Transformer Captions:")
    transformer_samples = generate_sample_captions(transformer_checkpoint, "transformer", 5, "val")

    return lstm_metrics, transformer_metrics

# Start training both models
print("Starting training...")
lstm_checkpoint, transformer_checkpoint = main_training_pipeline()

# Evaluate the trained models
print("Starting evaluation...")
lstm_metrics, trans_metrics = main_evaluation_pipeline(lstm_checkpoint, transformer_checkpoint)
