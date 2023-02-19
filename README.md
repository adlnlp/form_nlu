# Form-NLU: Dataset for the Form Language Understanding
we introduce a new dataset for form structure understanding and key information extraction. The dataset would enable the interpretation of the form designer's specific intention and the alignment of user-written value on it. Our dataset also includes three form types: digital, printed, and handwritten, which cover diverse form appearances/layouts and deal with their noises. In this repository, we will provide detailed baseline model descriptions and experimental setups for ensuring our model and experiments are reproducable. 
## Baseline Model Description
### Baseline Description
Here we introduce the baselines adopted in our project.

**VisualBERT**: is a transformer-based pretrained model which can jointly learn the contextual vision and language representations of input text and image regions.

**LXMERT**: proposes a three transformer encoder-based pretrained vision language model on increasing the ability to learn vision and language semantic connections.

**M4C**: a multimodal transformer encoder-decoder architecture designed for visual question answering of which inputs contain representations of the question, OCR tokens, and detected image object features for iteratively decoding the answers from input OCR tokens or fixed answer space. 

### Baseline Setup
**VisualBERT, LXMERT**: we feed the key text content and extract 2048-d visual features of each segment from the Res5 layer of ResNet101 into pretrain vision-language-models to learn the relation between segments and key contents. The enhanced visual representation of each segment will be fed into a pointer network to acquire a score for each segment. A softmax function is applied to predict the counterpart value index of the input key.

**M4C**: we slightly change the input features and output layers of the original M4C models to suit Task B demands. Firstly, OCR-extracted text representations are replaced by BERT $[CLS]$ token features, fed into the transformer encoder with key text content and segment visual features. Additionally, we use a pointer network to replace the originally adopted dynamic pointer network on the top of the transformer decoder to predict the index of the corresponding value. Except for the above two aspects, we utilize the same transformer encoder-decoder structure as the original M4C model.
## Multi-aspect Features

## Implementation Details

## Dataset Loading and Samples
