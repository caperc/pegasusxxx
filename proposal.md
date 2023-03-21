

DTS311TC FINAL YEAR PROJECT


A  Fine-tone of PEGASUS-X for Light Long Text Summary Generation

Proposal Report




In Partial Fulfillment
of the Requirements for the Degree of
Bachelor of Engineering


Student Name	:	Ke Huang
Student ID	:	1931030
Supervisor	:	Li Huangkang



School of  AI and Advanced Computing
Xi’an Jiaotong-Liverpool University
November 2022
Abstract  
Even while big pre-trained Transformer models have shown to be quite effective at handling natural language problems, dealing with lengthy input sequences remains a difficult task. Summarizing long inputs is one such job, when the length of the input surpasses the maximum input range of the majority of pre-trained models. 
We study how PEGASUS-X structural alterations and pre-training paradigms can best adapt pre-trained transformers to the summary of lengthy inputs through a comprehensive set of tests. Performance and efficiency are well-balanced using an interleaved, block-local transformer with a global encoder tag, and the performance of downstream summarization for lengthy sequences is considerably enhanced by an extra pre-training step. Building on PEGASUS-X, a PEGASUS model extension that can handle inputs with up to 16K tokens and additional long-input pretraining, PEGASUS-X achieves robust performance in long-input summarization tasks comparable to larger models with only a small number of additional parameters and without the need for parallel model training. Based on PEGASUS-X, we look for a possible solution that has good performance without consuming too much computational resources.
 



Contents
Abstract	ii
Contents	iii
1 Introduction	1
1.1 Introduction	1
1.1.1 Background	1
1.1.2 Extractive summarization	2
1.1.3 Generative summarization	2
1.2 Scope and Objectives	3
1.2.1 Computing resource challenges	3
1.2.2 Long text or document level relationship extraction	3
2 Literature Review	4
3 Project Plan	7
3.1 Proposed Solution / Methodology	7
3.2 Experimental Design	9
3.2.1 Dataset	9
3.2.2 Evaluation Indexes	9
3.3 Expected Results	9
3.4 Progress Analysis and Gantt Chart	10
3.4.1 Work Breakdown Structure	10
3.4.2 Critical milestones and risk analysis	11
4 Conclusion	12
References	13



1Introduction 
Introduction
1.1.1Background 
With the explosive growth of textual information in recent years, people have access to huge amount of textual information every day, such as news, reports, papers, blogs, chats, etc.. Extracting important contents from the large amount of text information has become a widespread and urgent need. Text summarization provides an efficient solution. According to Radev's definition [1], the summarization is "a text taken from one or more texts that incorporates key information from the original text and is no more than half the length of the full text.". Automated text summarization aims to automatically produce concise, fluent summaries that retain key information by machine. As one of the main directions of text generation tasks, it is essentially an information compression technique. Automatic text summarization has several application scenarios, including the development of automatic reports, news headlines, and search result previews. In addition, computerized text summarizing can benefit subsequent processes. Despite the huge demand for automatic text summarization, the development of this field has been slow. Generating summaries can be a challenging task for computers. Generating a qualified abstract from one or more texts requires a computer to read the original text and then understand its content and prioritize the content, crop and splice the content, and finally produce a short, fluent text. Therefore, automatic text summarization needs to rely on theories related to natural language processing/comprehension, which is one of the important research directions in recent years. Text summarization can be divided into single-document summarization and multi-document summarization according to the input type. Single-document summarization generates a summary from a given document, and multi-document summarization generates a summary from a given set of subject-related documents. 
According on the output technique, text summarizing is often separated into extractive summarization and abstractive summarization.
1.1.2Extractive summarization
Extractive summarization determine the important sentences in the original text and extract these sentences into an abstract. The generative approach applies sophisticated natural language processing methods to create more compact summaries by paraphrasing,  synonym substitution, and sentence abbreviation. 
1.1.3Generative summarization 
Generative summarization approaches are closer to the process of human abstracting than extractive approaches. Historically, extractive has usually outperformed generative. Along with the emergence and research of deep neural networks, the development of generative text summarization based on neural networks has been fast and has shown 
positive results. 


Figure 1: Example of Text Summarization for generated summary and extractive summary.

Scope and Objectives
1.2.1Computing resource challenges
The input sequence length of the majority of frequently researched summary tasks is often less than 512 symbols, which is shorter on average than the Transformer language model. This field needs to complete harder summary jobs with longer input lengths as model processing language advances. It is difficult to complete these lengthier summarization jobs due to the quadratic growth of the memory needs and computation for the attention mechanism in Transformers. To overcome this restriction, numerous memory- and computer-efficient Transformers variations have been developed such as Longformer [3], Big Bird [4] and FAVOR+ [5]. However, models are frequently pretrained on short sequence inputs and only be modified to handle long sequences when fine-tuning on a downstream task, which may be unsatisfactory even when including efficient Transformer topologies that provide roughly linear memory scaling with input sequences. 
1.2.2Long text or document level relationship extraction
Classical relationship extraction tends to focus on a single sentence and simply tries to uncover the entity relationships within each sentence. Documents often refer to many entities that embody complex cross-logical relationships, and extracting relationships from complex multi-sentence scenarios requires reading, memorizing, and reasoning to discover the relationship facts between multiple sentences.
In real scenarios such as medical and financial documents, there are many relationship facts embedded in entity pairs of different sentences in the document, and there are often complex interrelationships between multiple entities in the document. Compared to sentence level, the text in document level connection extraction is significantly lengthier and contains a greater number of entities, making document level relationship extraction more complex. 
(1) How to effectively model the multi-granularity information of entities, which mainly includes 2 points. The possibility of the same entity appearing in multiple sentences (i.e., entity mentions in multiple sentences). The issue of entity referencing that naturally occurs in text writing.
(2) How to model complex semantic information within a document, which mainly involves multiple aspects of reasoning, such as logical reasoning, denotational reasoning, and common-sense knowledge reasoning.
An important enhancement in the area of long text summarization is the introduction of transformer models such as BERT [6] and GPT-3 [7]. These models can handle longer text input sequences in a single run and provide a new understanding of chunking algorithms. Past architectures (such as LSTM [8] or RNN [9]) were neither as efficient nor as accurate as these transformer-based models, which made long document summarization more difficult. Growth in understanding of how to construct and use chunking algorithms that maintain the structure of contextual information and reduce data discrepancies at runtime is also key.

2Literature Review
Existing large-scale summary datasets consist of relatively short documents. For example, the articles in the CNN/Daily Mail dataset [11] are on average about 600 words long.
PEGASUS [13] However, in the real world we often need to deal with long text or document-level data General-purpose language models such as BERT, GPT-3, and XLNet [12] have demonstrated their power, and they can cope with all kinds of NLP tasks, such as text generation and question and answer. When these models are fine-tuned for various language tasks, SOTA performance can be achieved. These NLP models are "generalists" and, while comprehensive, require fine-tuning for specific tasks, and the training data sets are very large and beyond the reach of the average institution. A team from Google Brain and Imperial College London recently built a system - Pre-training with Extracted Gap-sentences for Abstractive Summarization (PEGASUS) [13], using Google's Transformer architecture, and incorporates pre-training goals tailored to text summarization generation. The "PEGASUS" model is specifically built for machine generated summaries. It has achieved state-of-the-art results (SOTA) in 12 abstracting tasks covering news, science, stories, usage notes, emails, patents and legislative bills, and is included in ICML 2020. The "PEGASUS" model is trained with only 1000 samples to approach the level of human summarization, significantly reducing the need for supervised data and creating the possibility of low-cost use.

Figure 2 The base architecture of PEGASUS is a standard Transformer encoder-decode [13]

BERTSUM [14] is a simple variant of BERT. It is used for extracting text summarization, mainly to selectively extract sentences in the text as the final summary. The biggest problem of this task is how to obtain the vector of each sentence, and then use the vector for secondary classification to determine whether to leave or not. The original BERT model can only generate sentence vectors or sentence pairs of single sentences. So Liu[14] sentence processing method:
(1)Add [CLS] before and [SEP] after each sentence in the document, and then enter BERT. The corresponding position of each [CLS] is the sentence vector of each sentence. 
(2)In order to further increase the interaction between sentences, a Transformer Summary Layer is added to BERT. Only the vector of each [CLS] is input, and finally the output predicts whether the current sentence is retained, finetune.


Figure 3 The overview architecture of the BERTSUM model [14]
Longformer [3] Transformer-based models, as we saw in the drawbacks for BertSum, struggle to handle extremely lengthy input sequences because of their self-attention mechanism, which scales rapidly as the sequence length increases. The longformer attempts to address this by employing two types of attention mechanisms: local windowed attention and task-motivated attention, which scale linearly with sequence length.
This makes processing documents containing thousands of tokens in a single instance much simpler. For the summarizing task, we employ a variant of this termed Longformer Encoder-Decoder. 

(a) Full n2 attention        (b) Sliding window attention     (c) Dilated sliding window  (d) Global+sliding window
Figure 4 contrasting Longformer's attention pattern configuration with the full self-attention pattern. 



3Project Plan 
Proposed Solution / Methodology
Replicate PEGASUS-X [15] model experiments using the SCROLLS [18] (Standardized CompaRison Over Long Language Sequences) dataset. Explore lightweight long-text summary generation methods based on PEGASUS-X that save computational and memory resources.
Although it has been demonstrated that huge pre-trained Transformer models can handle natural language tasks successfully, managing lengthy sequential inputs continues to be a significant difficulty. Long input summaries are one such job, where the inputs exceed the maximum input context of the majority of pre-trained models. Phang, Zhao and Liu [15] from New York University and Google Brain Team study which model architectural changes and pre-training methods can best adapt the pre-trained Transformer to lengthy input summaries through a comprehensive set of trials. Phang discover that an interleaved block local transformer with global encoder tokens achieves a good compromise between performance and efficiency, and that a second pre-training step on lengthy sequences significantly enhances downstream digest performance. Based on our research, Phang present PEGASUS-X, an extension of the PEGASUS model that can handle inputs of up to 16 K tokens by adding additional long-input pretraining. 
PEGASUS-X, with just a few extra parameters, performs robustly on the long-input summarization task equivalent to that of much bigger models without the need for parallel training models. 

Figure 5 Model performance on SCROLLS summary tasks  [15] 
We will study structural adjustments to the local and global local encoder models in light of their outstanding performance. The interleaving of local attention blocks is presented first. Block-like local attention is distinct from sliding window attention in that the mark may only perceive other markings inside the same block. If the input markers are divided into identical blocks at each layer, no information is communicated between the blocks inside the encoder. To address this issue, a modest structural adjustment is made, specifically, the placement of blocks on alternate layers is randomized. Figure 6 displays one scenario. Particularly, we stagger the attention block by changing half of the block border every other layer: in practice, we achieve this by filling half of the attention block with hidden representations on both sides and shielding them accordingly. 

(a) Block-local attention	        (b) Block-local attention with staggered blocks  [15]
Figure 6 In block-local attention (a), the same block boundaries are utilized throughout all levels, prohibiting the exchange of information across blocks. Staggering the block borders (b) by moving the boundaries every other layer enables cross-block interactions with minimally increased cost or complexity. 


Experimental Design
3.2.1Dataset 
Comparing encoder architecture Transformer, BigBird and Performer with Local and Global-Local encoder on short (XSUM [16], CNN/DM [11]) and long (arXiv [17], GovReport[19]) summarization tasks.
3.2.2Evaluation Indexes
ROUGE-1, Rouge-2, ROUGE-SU4, ROUGE-L [10]
A set of criteria called Rouge (Recall-Oriented Understudy for Gisting Evaluation) is used to assess automatic abstracts and machine translations. By contrasting them with a set of reference abstracts (often manually prepared), it determines the "similarity" between machine generated abstracts or translations and reference abstracts. 

Expected Results
We hope to achieve state of the art results for long text summary generation job on the selected dataset by improving the attention mechanism while maintaining the lightness of the model and reducing the use of parameters. Furthermore, the expectation is that the model will have generally excellent performance in a wider range of applications.

Progress Analysis and Gantt Chart
3.4.1Work Breakdown Structure
The work was brokedown into the following parts:
Time checkpoints	Works
Proposal	Dataset selection
	Algrithm
	Proposal
Dissertation	Hardware and software preparation
	Pretraining and Fine-tuning
	Experiments
	Analysis
	Report
Viva	Viva


The Gantt chart timeline for work plan:

Figure 7 Gantt chart for work plan
3.4.2Critical milestones and risk analysis
Important milestones in this project are：
Propose experimental design scheme
Lab environment deployment
Replicate existing algorithms
Propose a specific solution for model improvement
Improve existing work and obtain SOTA performance
Summarize experimental conclusions

First, the experimental design scheme has been given and will not be problematic. In the process of reproducing the existing algorithm, we may encounter problems such as the experimental environment cannot be deployed and the computational resources are insufficient. This problem has been considered in the design phase of the experimental scheme. If problems arise in the deployment phase of the lab environment, our backup solution is to choose Google Cloud Server for deployment. Since our computational resources are limited, the lightweight but efficient PIGASUS-X model is used. It is theoretically feasible to propose a better algorithmic model or to obtain SOTA results in the experiments at a later stage. However, the actual feasibility can be judged only after the experiments.


4Conclusion 
In our study, we examine a number of proposed enhancements to enable the Transformer model to efficiently and cheaply handle lengthy inputs in long text summarizing jobs. We discovered a simple and successful technique to expand the Transformer for short inputs to handle long-input summaries via much experimentation. We intend to attain state-of-the-art outcomes for long text summary generation on the selected datasets arXiv [17] and GovReport[19] by enhancing the attention mechanism while keeping the model's lightness and decreasing parameter usage. Besides, Both pre-training long input models from scratch and extending short sequence models that have previously been pre-trained can be used to expand models to handle lengthy input sequences in areas other than summarization. In conclusion, achieving our research goal will provide a cutting-edge model with low consumption of computing resources and high performance of application to deal with long text summary generation work.

Word count： 2228


References
[1] D. R. Radev, E. Hovy, K. McKeown. Introduction to the Special Issue on Summarization. Computational Linguistics 2002; 28 (4): 399–408. doi: https://doi.org/10.1162/089120102762671927
[2] M. Payne. 4 Powerful Long Text Summarization Methods With Real Examplese [Online]. Available: URL https://www.width.ai/post/4-long-text-summarization-methods (accessed Nov. 10, 2022).
[3] I. Beltagy, M. E. Peters, and A. Cohan, ‘Longformer: The Long-Document Transformer’. arXiv, Dec. 02, 2020. doi: 10.48550/arXiv.2004.05150.
[4] M. Zaheer et al., ‘Big Bird: Transformers for Longer Sequences’, in Advances in Neural Information Processing Systems, 2020, vol. 33, pp. 17283–17297. Accessed: Nov. 11, 2022. [Online]. Available: https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html
[5] K. Choromanski et al., ‘Rethinking Attention with Performers’. arXiv, Sep. 30, 2020. doi: 10.48550/arXiv.2009.14794.
[6] S. Alaparthi and M. Mishra, ‘Bidirectional Encoder Representations from Transformers (BERT): A sentiment analysis odyssey’. arXiv, Jul. 02, 2020. doi: 10.48550/arXiv.2007.01127.
[7] T. B. Brown et al., ‘Language Models are Few-Shot Learners’. arXiv, Jul. 22, 2020. doi: 10.48550/arXiv.2005.14165.
[8] S. Hochreiter and J. Schmidhuber, ‘Long Short-term Memory’, Neural computation, vol. 9, pp. 1735–80, Dec. 1997, doi: 10.1162/neco.1997.9.8.1735.
[9] W. Zaremba, I. Sutskever, and O. Vinyals, ‘Recurrent Neural Network Regularization’. arXiv, Feb. 19, 2015. Accessed: Nov. 11, 2022. [Online]. Available: http://arxiv.org/abs/1409.2329
[10] C.-Y. Lin, ‘ROUGE: A Package for Automatic Evaluation of summaries’, presented at the Proceedings of the ACL Workshop: Text Summarization Braches Out 2004, Jan. 2004, p. 10.
[11] Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems. pages 16931701.
[12] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, and Q. V. Le, ‘XLNet: Generalized Autoregressive Pretraining for Language Understanding’. arXiv, Jan. 02, 2020. doi: 10.48550/arXiv.1906.08237.
[13] J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu, ‘PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization’. arXiv, Jul. 10, 2020. doi: 10.48550/arXiv.1912.08777.
[14] Y. Liu, ‘Fine-tune BERT for Extractive Summarization’. arXiv, Sep. 05, 2019. doi: 10.48550/arXiv.1903.10318.
[15] J. Phang, Y. Zhao, and P. J. Liu, ‘Investigating Efficiently Extending Transformers for Long Input Summarization’. arXiv, Aug. 08, 2022. doi: 10.48550/arXiv.2208.04347.
[16] S. Narayan, S. B. Cohen, and M. Lapata, ‘Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization’. arXiv, Aug. 27, 2018. Accessed: Nov. 11, 2022. [Online]. Available: http://arxiv.org/abs/1808.08745
[17] M. Chen, Z. Chu, S. Wiseman, and K. Gimpel, ‘SummScreen: A Dataset for Abstractive Screenplay Summarization’, in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 2022, pp. 8602–8615. doi: 10.18653/v1/2022.acl-long.589.
[18] U. Shaham et al., ‘SCROLLS: Standardized CompaRison Over Long Language Sequences’. arXiv, Oct. 11, 2022. doi: 10.48550/arXiv.2201.03533.
[19] L. Huang, S. Cao, N. Parulian, H. Ji, and L. Wang, ‘Efficient Attentions for Long Document Summarization’. arXiv, Apr. 11, 2021. doi: 10.48550/arXiv.2104.02112.
