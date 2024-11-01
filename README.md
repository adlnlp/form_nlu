# Form-NLU: Dataset for the Form Natural Language Understanding

## Form-NLU Description
This repository contains code for the paper [Form-NLU: Dataset for the Form Natural Language Understanding](https://dl.acm.org/doi/abs/10.1145/3539618.3591886)
__<p align="center">Yihao Ding, Siqu Long, Jiabin Huang, Kaixun Ren, Xingxiang Luo, Hyunsuk Chung, Soyeon Caren Han</p>__

We hold **The Second Competition on Visually Rich Document Intelligence and Understanding (VRDIU)** using this dataset in conjunction with the **33rd International Joint Conference on Artificial Intelligence (IJCAI 2024)** in **Jeju, Korea**. The competition consists of <a href='https://www.kaggle.com/competitions/dociu2024-form-nlu?rvi=1'>Track A</a> and <a href='https://www.kaggle.com/competitions/dociu-form-nlu-2024-track-2/data'>Track B</a>.

__<p align="center">University of Sydney and University of Western Australia</p>__
<p align="center"><img src="images/task_definition_v2.jpg" width="750" /></p>

We introduce a new dataset to understand form structure and extract key information. This repository will provide detailed baseline model descriptions and experimental setups to ensure our model and experiments are reproducible. As well as we will offer a colab link to show how to download and use our dataset for corresponding tasks.   Currently, we will only provide a few samples of our Form-NLU. We will update the GitHub and a [colab tutorial](https://colab.research.google.com/drive/1m399VuMHU3zKvXQdtZAWAediPUE8hhQc) to ensure our model and dataset are publicly available after our paper is published. 

## Link to the Datasets
### Task A Dataset Link
For layout analysis, we provide the annotated coco files for the training and validation dataset with the document images. Please click [here](https://drive.google.com/file/d/1tVFb9ciMaQJ4hvmTnY53A59Z8qoddhCv/view?usp=drive_link) to get the dataset for task A.
As we will hold a competition on this dataset, we will only provide the images for three test sets and release the ground truth annotation results after the competition round. To ensure users can get the test results, we will launch the competition website soon to provide the user can upload their prediction results to get the evaluation results. 

### Task B Dataset Link
Please refer to our competition.

## Baseline Model Description
### Baseline Description
Here we introduce the baselines adopted in our project.

**VisualBERT**: is a transformer-based pretrained model which can jointly learn the contextual vision and language representations of input text and image regions.

**LXMERT**: proposes a three-transformer encoder-based pretrained vision language model to increase the ability to learn vision and language semantic connections.

**M4C**: a multimodal transformer encoder-decoder architecture designed for visual question answering of which inputs contain representations of the question, OCR tokens, and detected image object features for iteratively decoding the answers from input OCR tokens or fixed answer space. 

### Baseline Setup
**VisualBERT, LXMERT**: we feed the key text content and extract 2048-d visual features of each segment from the Res5 layer of ResNet101 into pretrain vision-language models to learn the relation between segments and key contents. The enhanced visual representation of each segment will be fed into a pointer network to acquire a score for each segment. A softmax function is applied to predict the counterpart value index of the input key.

**M4C**: we slightly change the input features and output layers of the original M4C models to suit Task B demands. Firstly, OCR-extracted text representations are replaced by BERT *[CLS]* token features, fed into the transformer encoder with key text content and segment visual features. Additionally, we use a pointer network to replace the originally adopted dynamic pointer network on the top of the transformer decoder to predict the index of the corresponding value. Except for the above two aspects, we utilize the same transformer encoder-decoder structure as the original M4C model.
## Multi-aspect Features
**Visual Feature** (V): is used to numerically represent the appearance features of each input RoI by using 2048-d vectors extracted from the Res5-layer of ResNet-101 pretrained on ImageNet. 

**Textual Features** (S): aims to represent the semantic meaning of RoI's text content (acquired from OCR or PDFminer). We extract *[CLS]* token representations by feeding the tokenized text into pretrained *BERT-base-uncased* to generate a 768-d textual feature for each RoI. 

**Positional Features** (P): aims to comprehensively physical layout structure of form appearance. It is normalized bounding box coordinates $(\frac{x_i}{W},\frac{y_i}{H},\frac{w_i}{W},\frac{h_i}{H})$ of each input RoI where $W$ and $H$ are width and height of corresponding input page. 

**Density Features** (D): could effectively enhance segment representations which have been demonstrated by [DocGCN](https://github.com/adlnlp/doc_gcn). We use normalized number of characters $norm\_{num^{char}\_i} = \frac{num^{char}\_i}{max(num^{char})}$, where $max(num^{char})$ represents the maximum number of characters among all segments. Additionally, we also use $den^{char}_i = \frac{num^{char}_i}{(w_i \times h_i)}$ to represent the text density which is concatenated with $norm\_num^{char}_i$ to represent $Den_i$ of i-th input $r_i$.

**Gap Distance Features** (G): aims to represent the spatial relationship between neighbouring entities, which is the gap distance between $r_i$ with four directions nearest RoIs (up, down, left, right), represented by $G_i = [gd_i^{u}, gd_i^{d}, gd_i^l,gd_i^r]$. Consequently, similar to $P$, we also use $W$ and $H$ to get the normalized $G_i^{norm} = [\frac{gd_i^{u}}{H}, \frac{gd_i^{d}}{H}, \frac{gd_i^l}{W},\frac{gd_i^r}{W}]$.

Regarding Task B, we employ various approaches to encode the vision and language features. Firstly, all Task B adopted baselines use pretrained BERT to encode key textual content. Moreover, for the visual aspect, VisualBERT, LXMERT, and M4C models utilize 2048-d features extracted from the Res5 layer of ResNet101. The maximum number of input key text tokens and the number of segments on each page are all defined as 50 and 41, respectively. Task A and B experiments are conducted on Tesla V100-SXM2 with CUDA11.2.

## Case Study Setup
We conduct a transfer learning case study on the FUNSD dataset. We manually select 200 valuable question-answer pairs from the original FUNSD dataset. The FUNSD subset of the case study will be released after this paper is accepted. Then we use the trained model on our dataset to test whether the model can predict the correct answer of input question-answer pairs of the FUNSD subset. When we feed the QA pair into the model, we replace the original question text with fixed keys used in Task B. If any key in 12 key sets can predict the correct question answer, we will count this sample as a True Positive sample for calculating accuracy.
## Dataset Loading and Samples
We provide an [colab notebook](https://colab.research.google.com/drive/1m399VuMHU3zKvXQdtZAWAediPUE8hhQc) to ensure researchers in both academic and industrial areas can access our Form-NLU.

## Citation
```
@inproceedings{ding2023form,
  title={Form-NLU: Dataset for the Form Natural Language Understanding},
  author={Ding, Yihao and Long, Siqu and Huang, Jiabin and Ren, Kaixuan and Luo, Xingxiang and Chung, Hyunsuk and Han, Soyeon Caren},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2807--2816},
  year={2023}
}

@inproceedings{vrdiu2024form,
  title={The Second Competition on Visually Rich Document Intelligence and Understanding},
  author={Han, Soyeon Caren and Ding, Yihao and Li, Yan and Cagliero, Luca and Park, Seong-Bae and Mitra, Prasenjit},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
  year={2024}
}
```
