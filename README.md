# Form-NLU: Dataset for the Form Language Understanding
We introduce a new dataset for form structure understanding and key information extraction. In this repository, we will provide detailed baseline model descriptions and experimental setups for ensuring our model and experiments are reproducable. 
## Baseline Model Description
### Baseline Description
Here we introduce the baselines adopted in our project.

**VisualBERT**: is a transformer-based pretrained model which can jointly learn the contextual vision and language representations of input text and image regions.

**LXMERT**: proposes a three transformer encoder-based pretrained vision language model on increasing the ability to learn vision and language semantic connections.

**M4C**: a multimodal transformer encoder-decoder architecture designed for visual question answering of which inputs contain representations of the question, OCR tokens, and detected image object features for iteratively decoding the answers from input OCR tokens or fixed answer space. 

### Baseline Setup
**VisualBERT, LXMERT**: we feed the key text content and extract 2048-d visual features of each segment from the Res5 layer of ResNet101 into pretrain vision-language-models to learn the relation between segments and key contents. The enhanced visual representation of each segment will be fed into a pointer network to acquire a score for each segment. A softmax function is applied to predict the counterpart value index of the input key.

**M4C**: we slightly change the input features and output layers of the original M4C models to suit Task B demands. Firstly, OCR-extracted text representations are replaced by BERT *[CLS]* token features, fed into the transformer encoder with key text content and segment visual features. Additionally, we use a pointer network to replace the originally adopted dynamic pointer network on the top of the transformer decoder to predict the index of the corresponding value. Except for the above two aspects, we utilize the same transformer encoder-decoder structure as the original M4C model.
## Multi-aspect Features
**Visual Feature** (V): is used to numerically represent the appearance features of each input RoI by using 2048-d vectors extracted from the Res5-layer of ResNet-101 pretrained on ImageNet. 

**Textual Features** (S): aims to represent the semantic meaning of RoI's text content (acquired from OCR or PDFminer). We extract *[CLS]* token representations by feeding the tokenized text into pretrained *BERT-base-uncased* to generate a 768-d textual feature for each RoI. 

**Positional Features** (P): aims to comprehensively physical layout structure of form appearance. It is normalized bounding box coordinates $(\frac{x_i}{W},\frac{y_i}{H},\frac{w_i}{W},\frac{h_i}{H})$ of each input RoI where $W$ and $H$ are width and height of corresponding input page. 

**Density Features** (D): could effectively enhance segment representations which have been demonstrated by [DocGCN](https://github.com/adlnlp/doc_gcn/edit/main). We use normalized number of characters $norm\_{num^{char}\_i} = \frac{num^{char}\_i}{max(num^{char})}$, where $max(num^{char})$ represents the maximum number of characters among all segments. Additionally, we also use $den^{char}_i = \frac{num^{char}_i}{(w_i \times h_i)}$ to represent the text density which is concatenated with $norm\_num^{char}_i$ to represent $Den_i$ of i-th input $r_i$.

**Gap Distance Features** (G): aims to represent the spatial relationship between neighbouring entities, which is the gap distance between $r_i$ with four directions nearest RoIs (up, down, left, right), represented by $G_i = [gd_i^{u}, gd_i^{d}, gd_i^l,gd_i^r]$. Consequently, similar to $P$, we also use $W$ and $H$ to get the normalized $G_i^{norm} = [\frac{gd_i^{u}}{H}, \frac{gd_i^{d}}{H}, \frac{gd_i^l}{W},\frac{gd_i^r}{W}]$.
## Implementation Details
For Task A, we fine-tune Faster-RCNN and Mask-RCNN models with two backbones\footnote{Faster-RCNN backbones are faster\_rcnn\_R\_50\_FPN\_3x, faster\_rcnn\_R\_101\_FPN\_3x and Mask-RCNNs are mask\_rcnn\_R\_50\_FPN\_3x, mask\_rcnn\_R\_101\_FPN\_3x} respectively on our dataset based on Detectron2 platform. We set 5000, 128, and 0.02 as the maximum iteration times, batch size and base learning rate, and other setups are the same as [Detectorn2 official tutorial]( https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5). 

Regarding Task B, we employ various approaches to encode the vision and language features. Firstly, all Task B adopted baselines use pretrained BERT to encode key textual content. Moreover, for the visual aspect, VisualBERT, LXMERT, and M4C models utilize 2048-d features extracted from the Res5 layer of ResNet101. The maximum number of input key text tokens and the number of segments on each page are all defined as 50 and 41, respectively. Task A and B experiments are conducted on 51 GB Tesla V100-SXM2 with CUDA11.2.
## Dataset Loading and Samples
For 
