# HoTS Model

## Convolution and Recurrent Model

Convolutional neural network (CNN) is devised to capture local patterns.
Thus, CNN is used for image process such as object detection and image classification, outperforming previous methods.
Recently, CNN on protein sequence is used to predict drug-target interaction.<sup>[DeepDTA](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245),[DeepConv-DTI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007129)</sup>
CNN is useful to capture local residue patterns interacting chemical ligands, creating latent representation from raw protein sequence.
In previous studies for DTI prediction using CNN, they recruited only CNN. Therefore, global inter-residue dependencies  are not regarded in model.
This limitation is pointed out some studies.<sup>[3](https://www.liebertpub.com/doi/abs/10.1089/cmb.2019.0193)</sup>

On the other hand, recurrent neural network (RNN) is used for protein, because protein is sequence of amino acids, which is most adequate data structure of RNN.
For example, SPIDER3 used stacked bidirectional long short term memory models (LSTM) to predict backbone angles, contact numbers abd solvent accessibility. <sup>[SPIDER3](https://academic.oup.com/bioinformatics/article/33/18/2842/3738544j)</sup>  

Because both CNN and RNN are used for protein. Neural network models combining CNN and RNN are developed. 

For example, to ensure regarding global residue interaction which is unable for CNN, stacked bidirectional gated rectifier unit (BGRU) is used on CNN<sup>[5](https://pdfs.semanticscholar.org/586e/1cd8cffb9fbf50ab70a7e65eb507b083db3f.pdf)</sup>.
However, there is limitation that RNN is unable to cope with extremely long sequences for global inter-residue dependencies.

## Object Detection

Object detection is technology in field of computer vision, detecting some object from whole image and classify objects.
Nowadays, CNN on image, which captures local patterns, is most suitable model to detect and distinguish object from background, outperforming previous methods.

For example, model called YOLO (You only look once), single-shot detector, use simple CNN model to retrieve object from image.<sup>[YOLOv1](https://pjreddie.com/media/files/papers/yolo_1.pdf),[YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf)</sup>


## Loss

### Focal loss

**Focal loss** is developed for dataset whose classes are very imbalanced.
Especially, in object detection problem, the number of object (positive) is much less than background (negative class) in label for model.
Therefore, loss for imbalanced dataset is needed to train model more properly.

Focal loss is constructed from binary cross-entropy ($CE$)

$$CE(x,y) = \left\{\begin{matrix}
-log(p)&& \text{if}\ y=1\\-log(1-p)&&\text{otherwise}
\end{matrix}\right.$$

Then Focal loss $FL$ is defined as 

$$FL(p_t) = -(1-p_{t})^{\gamma}\log(p_{t})$$

Exponential gamma term is added to binary entropy, which enables loss function dynamically learn imbalanced data.
In more detail, for large negative data, usually most of them are easier to predict as negative.
Focal loss whose gamma is larger than 1 gives less loss than binary cross-entropy for easy negative case, concentrating on train harder negative case.
As a result, it prevents model to have biased by a large number of negative data.
The percentage of binding sites and binding regions is much smaller than percentage of non-binding residues.
We expect to learn imbalanced class in unbiased way for model by using focal loss. 