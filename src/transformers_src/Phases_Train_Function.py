import wandb
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.distributions import MultivariateNormal
from src.Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word
from src.Inference import generate_sentence_ids
from src.NLPLPIPS import NLPLPIPS

def TwoPhaseTraining(emb_epochs, decode_epochs, emb_optimizer, decode_optimizer, model, train_dataset, max_seq_len, vocab, name, batch_size, lr, N, device, wandb_log):
    if wandb_log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project="Final_Metrics",
            name=f"{name}_bs{batch_size}_lr{lr}_N{N}",
            config={
                "epochs": emb_epochs+decode_epochs,
                "batch_size": batch_size,
            })
    decoding_criterion = nn.CrossEntropyLoss()
    nlp_lpips = NLPLPIPS()
    train_num_samples = len(train_dataset)
    smoothie = SmoothingFunction().method1
    for epoch in range(emb_epochs + decode_epochs):
        # Phase 1: Train the normalizing flow
        if epoch < emb_epochs:

            if epoch == 0:
                print('Embedding Training')

            model.FlowEmbedding.train()
            model.Transformer.eval()
            total_emb_loss = 0

            for batch_start in range(0, train_num_samples, batch_size):
                #print(batch_start)
                batch_end = min(batch_start + batch_size, train_num_samples)
                batch = train_dataset[batch_start:batch_end]

                src, trg = batch
                src, trg = src.to(device), trg.to(device)

                z, emb_loss = model.FlowEmbedding(src.float())

                total_emb_loss += emb_loss
                emb_optimizer.zero_grad()
                emb_loss.backward()
                emb_optimizer.step()

            epoch_emb_loss = total_emb_loss/train_num_samples

            print(f"Embedding Epoch {epoch + 1}/{emb_epochs} Embedding Loss: {epoch_emb_loss:.4f} ")

        else:
            if epoch == emb_epochs:
                print('Transformer Training')
            model.FlowEmbedding.eval()  # Set flow to evaluation mode to fix its parameters
            model.Transformer.train()  # Set transformer to training mode

            total_loss_train = 0.0
            total_bleu_train = 0
            total_word_overlap_train = 0
            total_lpips_train = 0
            total_jaccard_train = 0
            total_samples_training = 0

            for batch_start in range(0, train_num_samples, batch_size):
                batch_end = min(batch_start + batch_size, train_num_samples)
                batch = train_dataset[batch_start:batch_end]

                src, trg = batch
                src, trg = src.to(device), trg.to(device)

                embeddings, _ = model.FlowEmbedding(src.float())
                output, softmax_output = model(embeddings, trg.float())

                # Compute loss
                decode_loss = decoding_criterion(output.reshape(-1, output.size(-1)), trg.reshape(-1))

                # Backward pass
                decode_loss.backward()

                # Update model parameters
                decode_optimizer.step()

                total_loss_train += loss.item()

                for i in range(trg.size(0)):
                # Get the reference sentence
                    ref = trg[i].tolist()
                    ref_sentence = detokenize(ref, vocab)
                    ref_sentence = remove_word(ref_sentence, '[PAD]')
                    ref_sentence = remove_word(ref_sentence, '[SOS]')
                    ref_sentence = remove_word(ref_sentence, '[EOS]')


                    # Get the hypothesis
                    hyp = softmax_output.argmax(dim=-1)[i].tolist()
                    hyp_sentence = detokenize(hyp, vocab)
                    hyp_sentence = remove_word(hyp_sentence, '[PAD]')
                    hyp_sentence = remove_word(hyp_sentence, '[SOS]')
                    hyp_sentence = remove_word(hyp_sentence, '[EOS]')

                    # Calculate BLEU score for this sentence pair
                    bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
                    total_bleu_train += bleu_score

                    # Calculate word overlap
                    overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
                    total_word_overlap_train += overlap_score

                    # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
                    lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
                    total_lpips_train += lpips_dist.item()

                    # Calculate Jaccard similarity
                    jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
                    total_jaccard_train += jaccard_score

            epoch_loss = total_loss_train / total_samples_training
            epoch_bleu = total_bleu_train / total_samples_training
            epoch_word_overlap = total_word_overlap_train / total_samples_training
            epoch_lpips = total_lpips_train / total_samples_training
            epoch_jaccard = total_jaccard_train / total_samples_training

            if wandb_log:
                metrics = {"Train_loss": epoch_loss, "Train_Bleu": epoch_bleu, "Train_Word_Overlap": epoch_word_overlap, "Train NLPIPS": epoch_lpips, "Train Jaccard": epoch_jaccard}
            wandb.log(metrics)

            print(f"Transformer Epoch {epoch-emb_epochs + 1}/{decode_epochs} Loss: {epoch_loss:.4f} BLEU: {epoch_bleu:.4f} Word Overlap: {epoch_word_overlap:.4f}% Jaccard Similarity: {epoch_jaccard:.4f} Train LPIPS: {epoch_lpips:.4f}")

    print('Training Complete')
