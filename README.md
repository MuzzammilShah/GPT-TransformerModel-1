## Building GPT from Scratch - GPT 1

This project is an implementation of a GPT-style language model following Andrej Karpathy’s (iconic, if i may add) "Let’s Build GPT from Scratch" video. It walks through the key components of modern transformers, from a simple bigram model to a fully functional self-attention mechanism and multi-headed transformer blocks.  

### **Key Topics Covered**  
- **Baseline Model**: Bigram language modeling, loss calculation, and text generation.
- **Self-Attention**: Understanding matrix multiplications, softmax, and positional encodings. 
- **Transformer Architecture**: Multi-head self-attention, feedforward layers, residual connections, and layer normalization.
- **Scaling Up**: Dropout regularization, encoder vs. decoder architectures (only decoder block has been implemented, no encoder).

> [!Note]
> **Changes from the original video and Notes**
> - I've used a different and a bigger dataset for this, namely the 'Harry Potter Novels' collection. I found the raw dataset on kaggle (as 7 individual datasets) after which i had them merged and cleaned up seperately, so that the outputs can be a lot more cleaner. You may find the notebooks which I had implemented for that under the `additional-files` directory, so feel free to check that out.
> - This model is trained on 6 million characters (so ~6 million tokens)
> - The final output can be found in the file `generated.txt`.
> - I ran this model on a **NVIDIA GeForce GTX 1650** of my personal laptop with a decent amount of GPU memory and it took **approximately 90 minutes** to train and generate the final document.
> - I've also added breakdowns of the codes based on andrej's explainations and how much I understood so feel free to read them as well.

### **⭐Documentation**
For a better reading experience and detailed notes, visit my **[Road to GPT Documentation Site](https://muzzammilshah.github.io/Road-to-GPT/GPT-1/)**.