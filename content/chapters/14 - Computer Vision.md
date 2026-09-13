---
type: chapter
number: 14
order: 14
title: Computer Vision
book_pages: 592-689
pdf_pages: 632-729
status: reviewed
---
# Computer Vision

> **Ý chính trong một câu:** từ một ảnh, computer vision không chỉ hỏi “đây là gì?” mà còn học “ở đâu?”, “pixel nào thuộc vật nào?” và “làm sao biến đổi ảnh mà vẫn giữ điều cần nhận biết?”.

## Mục tiêu học tập

Học xong chương, bạn có thể:

- dùng augmentation và fine-tuning đúng giữa train/inference;
- biểu diễn, đổi hệ tọa độ và so sánh bounding boxes bằng IoU;
- giải thích anchors, multiscale detection, SSD, R-CNN/Fast/Faster/Mask R-CNN;
- chuẩn bị detection/segmentation datasets mà không làm lệch label;
- giải thích transposed convolution và fully convolutional network;
- mô tả style transfer bằng content/style/total-variation losses;
- tổ chức một image competition pipeline có validation và submission rõ ràng.

## Bản đồ chương

```text
ảnh + nhãn ít
 ├─ augmentation + fine-tuning → classification tốt hơn
 ├─ bounding boxes + anchors + IoU/NMS
 │      ├─ SSD: one-stage, multiscale
 │      └─ R-CNN family: proposals → classify/refine
 ├─ pixel labels → semantic segmentation
 │      └─ transposed conv → FCN → output cùng H×W
 └─ frozen CNN features → neural style transfer
```

## Bức tranh tổng quan

Image classification trả một nhãn cho cả ảnh. Object detection trả nhiều `(class, box)`. Semantic segmentation trả class cho từng pixel. Instance segmentation còn tách hai vật cùng class thành hai instances. Bốn mức output này quyết định label format, loss và model architecture.

| Task | Output điển hình | Ví dụ |
|---|---|---|
| Classification | `(batch, classes)` | ảnh có mèo hay chó |
| Detection | nhiều `(class, score, box)` | mèo ở box nào |
| Semantic segmentation | `(batch, classes, H, W)` | mỗi pixel là mèo/chó/nền |
| Instance segmentation | box + mask từng instance | tách riêng hai con chó |

<!-- pagebreak -->

## 14.1 Image Augmentation

{{term:image-augmentation|Image augmentation}} tạo các training examples giống nhưng không trùng từ ảnh có sẵn. Mục tiêu không chỉ “làm dataset to hơn”, mà buộc model bớt dựa vào tín hiệu ngẫu nhiên như vị trí tuyệt đối, độ sáng hay màu cụ thể.

### 14.1.1 Common Image Augmentation Methods

- `RandomHorizontalFlip`: thường hợp với vật thể tự nhiên; không luôn hợp với chữ/số hoặc biển báo có hướng.
- `RandomVerticalFlip`: hiếm hợp hơn; xe lật ngược không còn tự nhiên.
- `RandomResizedCrop`: crop vùng ngẫu nhiên rồi resize, làm đổi vị trí/kích thước vật.
- `ColorJitter`: đổi brightness, contrast, saturation, hue.
- `Compose`: ghép nhiều biến đổi.

```python
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])
```

**Code convention:** tách `train_transform` và `test_transform`; tên thể hiện vai trò, mỗi transform một dòng, randomness chỉ ở train. Với detection/segmentation, transform hình học phải cập nhật box/mask cùng ảnh.

### 14.1.2 Training with Image Augmentation

Sách huấn luyện ResNet-18 trên CIFAR-10 và tạo mỗi batch bằng augmentation ngẫu nhiên. Cùng một ảnh gốc có thể hiện ra khác nhau qua epoch. Khi inference, không dùng random transform để output có tính xác định (trừ khi chủ ý dùng test-time augmentation và gộp nhiều predictions).

### 14.1.3 Summary

- Augmentation tăng đa dạng quan sát và thường cải thiện generalization.
- Random augmentation dùng cho training, không mặc định dùng cho prediction.
- Nhiều transform có thể phối hợp, nhưng phải tôn trọng invariance thật của bài toán.

### 14.1.4 Exercises

1. **Bỏ augmentation:** training accuracy thường tăng nhanh hơn, test accuracy có thể thấp hơn; khoảng cách train-test lớn là bằng chứng hỗ trợ giảm overfitting, nhưng cần nhiều seed để kết luận.
2. **Ghép transform:** thử flip + crop + color jitter; transform quá mạnh có thể phá class semantics.
3. **Tìm transform khác:** rotation, affine, grayscale, perspective, erasing; với mỗi loại phải trả lời “label còn đúng không?”.

## 14.2 Fine-Tuning

{{term:transfer-learning|Transfer learning}} chuyển kiến thức từ source dataset lớn sang target dataset nhỏ. {{term:fine-tuning|Fine-tuning}} là cách phổ biến: copy kiến trúc/weights pretrained, thay output layer, rồi tiếp tục học.

### 14.2.1 Steps

1. Pretrain source model trên dataset lớn như ImageNet.
2. Tạo target model; copy mọi layer trừ output head.
3. Khởi tạo output head theo số target classes.
4. Fine-tune body với learning rate nhỏ; train head mới với learning rate lớn hơn.

Body đã có features cạnh, texture, hình dạng. Update quá lớn có thể phá representation tốt; head mới thì cần học nhanh từ đầu.

### 14.2.2 Hot Dog Recognition

Sách dùng dataset hot dog / not hot dog, normalize theo ImageNet, tải ResNet-18 pretrained và thay `fc` từ 1000 classes thành 2. Optimizer đặt learning rate của head khoảng 10 lần body.

```python
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

optimizer = torch.optim.SGD([
    {"params": model.parameters(), "lr": 5e-4},
    {"params": model.fc.parameters(), "lr": 5e-3},
], momentum=0.9)
```

Trong code thật, nhóm parameter ở trên bị lặp vì `model.parameters()` đã chứa `model.fc`; cần tách body/head rõ ràng:

```python
body_params = [
    parameter
    for name, parameter in model.named_parameters()
    if not name.startswith("fc.")
]
optimizer = torch.optim.SGD([
    {"params": body_params, "lr": 5e-4},
    {"params": model.fc.parameters(), "lr": 5e-3},
], momentum=0.9)
```

### 14.2.3 Summary

- Fine-tuning copy pretrained features, thay output layer và học tiếp trên target.
- Body dùng bước nhỏ; head mới có thể dùng bước lớn.

### 14.2.4 Exercises

1. Tăng learning rate tới khi validation dao động/giảm: đó là dấu hiệu phá weights tốt hoặc bước quá lớn.
2. So sánh pretrained với scratch bằng cùng split, seed và budget; nếu không, kết luận không công bằng.
3. Freeze body (`requires_grad=False`) nhanh và ít overfit hơn nhưng kém thích nghi nếu target khác ImageNet.
4. ImageNet có class hotdog: có thể dùng row weight tương ứng làm initialization cho logit hotdog, nhưng vẫn cần head hai lớp và calibration trên target.

<!-- pagebreak -->

## 14.3 Object Detection and Bounding Boxes

### 14.3.1 Bounding Boxes

{{term:object-detection|Object detection}} nhận biết cả class và vị trí. {{term:bounding-box|Bounding box}} thường có hai cách biểu diễn:

- corner: $(x_{min},y_{min},x_{max},y_{max})$;
- center: $(x_c,y_c,w,h)$.

Chuyển đổi:

$$x_c=(x_{min}+x_{max})/2,\quad w=x_{max}-x_{min}$$
$$y_c=(y_{min}+y_{max})/2,\quad h=y_{max}-y_{min}.$$

Tensor boxes có shape `(..., 4)` vì bốn số ở axis cuối mô tả một box. Tọa độ có thể theo pixel hoặc normalize về $[0,1]$; không được trộn hai convention.

```python
import torch


def corners_to_center(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (..., xmin, ymin, xmax, ymax) to (..., cx, cy, w, h)."""
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    return torch.stack(
        ((x_min + x_max) / 2, (y_min + y_max) / 2,
         x_max - x_min, y_max - y_min),
        dim=-1,
    )
```

### 14.3.2 Summary

- Detection dự đoán mọi vật quan tâm và vị trí của chúng.
- Box là approximation hình chữ nhật; corner/center formats đổi qua lại được.

### 14.3.3 Exercises

1. Tự gắn box một ảnh: box thường tốn thời gian hơn class vì phải quyết định biên và xử lý che khuất.
2. Axis cuối luôn dài 4 vì mỗi box cần đúng bốn bậc tự do theo convention đã chọn.

## 14.4 Anchor Boxes

{{term:anchor-box|Anchor boxes}} là các box ứng viên nhiều scale/aspect ratio, đặt quanh các vị trí trên feature map. Model không dự đoán box từ khoảng không; nó phân loại anchor và dự đoán offset để sửa anchor.

### 14.4.1 Generating Multiple Anchor Boxes

Với ảnh cao $h$, rộng $w$, scale $s$ và aspect ratio $r=w_b/h_b$:

$$w_b=ws\sqrt r,\qquad h_b=hs/\sqrt r.$$

Nếu có $n$ scales và $m$ ratios, sách dùng các kết hợp $(s_1,r_j)$ và $(s_i,r_1)$ để mỗi vị trí có $n+m-1$ anchors, tránh $nm$ boxes quá nhiều.

### 14.4.2 Intersection over Union (IoU)

{{term:intersection-over-union|Intersection over Union}} đo độ chồng:

$$\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

IoU = 0 khi không giao, = 1 khi trùng hẳn. Nó không quan tâm class và không phải khoảng cách hình học thông thường.

### 14.4.3 Labeling Anchor Boxes in Training Data

Mỗi ground-truth phải có ít nhất một anchor được gán. Các anchors còn lại có IoU vượt threshold cũng được gán; thấp thì background. Label gồm:

- class (background thường là 0, object classes dịch lên 1);
- mask cho offsets;
- offsets chuẩn hóa giữa anchor $a$ và ground truth $b$:

$$\Delta x=10\frac{x_b-x_a}{w_a},\quad \Delta y=10\frac{y_b-y_a}{h_a},$$
$$\Delta w=5\log\frac{w_b}{w_a},\quad \Delta h=5\log\frac{h_b}{h_a}.$$

### 14.4.4 Predicting Bounding Boxes with Non-Maximum Suppression

Inference đảo offsets để ra predicted boxes. Nhiều anchors có thể cùng phát hiện một vật. {{term:non-maximum-suppression|Non-maximum suppression}} (NMS) greedily:

1. giữ box score cao nhất;
2. bỏ boxes cùng class có IoU với nó vượt threshold;
3. lặp với phần còn lại.

NMS có thể bỏ một box hữu ích khi hai vật thật ở sát nhau. Soft-NMS giảm score thay vì xóa cứng.

### 14.4.5 Summary

- Anchors lấy mẫu vùng theo vị trí, scale, ratio.
- IoU đo overlap; labels gồm class và offset.
- NMS rút nhiều predictions trùng thành output gọn.

### 14.4.6 Exercises

1. Tăng sizes tạo boxes lớn; thêm ratios tạo boxes dài/ngang. Vẽ trước khi train.
2. Hai box IoU 0.5 cần intersection bằng một nửa union; với hai box vuông bằng nhau dịch ngang $d$, giải $(1-d)/(1+d)=0.5$ được $d=1/3$ cạnh.
3. Đổi anchors làm thay assignment, positive/negative balance và offsets.
4. Soft-NMS dùng $score\leftarrow score\cdot f(IoU)$ thay xóa.
5. NMS có thể học (learned NMS/set prediction), nhưng phức tạp hơn và cần dữ liệu/thiết kế permutation-invariant.

<!-- pagebreak -->

## 14.5 Multiscale Object Detection

### 14.5.1 Multiscale Anchor Boxes

Sinh 5 anchors ở mọi pixel ảnh 561×728 tạo hơn hai triệu anchors. Ta dùng feature maps:

- feature map lớn, nhiều vị trí → nhiều anchor nhỏ cho vật nhỏ;
- feature map nhỏ, ít vị trí → ít anchor lớn cho vật lớn.

Mỗi cell feature map tương ứng một {{term:receptive-field|receptive field}} trên ảnh. Prediction tại cell dùng thông tin vùng đó.

### 14.5.2 Multiscale Detection

CNN càng sâu cho feature map nhỏ hơn và features trừu tượng hơn. Detection head gắn ở nhiều scales để cân bằng localization của vật nhỏ với semantic richness cho vật lớn.

### 14.5.3 Summary

- Feature map shape quyết định tâm anchors được sample.
- Layerwise representations cho phép detection ở nhiều kích thước.

### 14.5.4 Exercises

1. Feature maps khác scale thường cũng khác mức abstraction vì đến từ độ sâu khác, nhưng scale và abstraction không phải cùng một khái niệm bắt buộc.
2. Với 4×4 map, tâm ở $(i+0.5)/4$; anchor đủ lớn sẽ chồng nhau tự nhiên.
3. Map `(1,c,h,w)` qua conv head thành `(1,a(q+1),h,w)` cho class và `(1,4a,h,w)` cho offsets; reshape để mỗi anchor là một row.

## 14.6 The Object Detection Dataset

Sách dùng dataset chuối nhỏ: 1000 ảnh tổng hợp, mỗi ảnh có một banana ở vị trí/kích thước khác nhau.

### 14.6.1 Downloading the Dataset

Dataset gồm images và CSV labels. Check checksum khi download; data là một phần của thí nghiệm, không phải bước phụ có thể bỏ qua.

### 14.6.2 Reading the Dataset

Mỗi label là `[class, xmin, ymin, xmax, ymax]`, tọa độ normalize theo 256. Batch image shape `(batch, 3, 256, 256)`; batch labels shape `(batch, 1, 5)` vì mỗi ảnh đúng một vật.

### 14.6.3 Demonstration

Luôn vẽ box ground truth lên vài ảnh sau transform. Đây là unit test trực quan cho coordinate convention và augmentation.

### 14.6.4 Summary

- Detection loader giống classification loader ở ảnh, khác ở label có boxes.
- Normalize boxes giúp độc lập hơn với kích thước pixel.

### 14.6.5 Exercises

1. Quan sát nhiều ảnh: box đổi theo vị trí/kích thước banana, class giữ 0.
2. Random crop phải crop cả box; nếu chỉ còn mảnh nhỏ của object, đặt ngưỡng visible area để giữ/bỏ và clip box vào ảnh mới.

<!-- pagebreak -->

## 14.7 Single Shot Multibox Detection

{{term:single-shot-multibox-detection|Single Shot Multibox Detection}} (SSD) là one-stage detector: một forward pass tạo anchors, class scores và box offsets ở nhiều scales.

### 14.7.1 Model

![SSD dùng base network và nhiều multiscale feature maps để dự đoán class và box](../assets/chapter-14/figure-14-7-1-ssd-architecture.png "Nguồn: didl.pdf, Figure 14.7.1, trang sách 631, trang PDF 671")

Base network trích features; mỗi block sau giảm $H,W$. Ở mỗi scale:

- class predictor là conv với $a(q+1)$ output channels;
- box predictor là conv với $4a$ output channels;
- predictions được permute/flatten rồi concatenate across scales.

TinySSD của sách có năm prediction scales. `forward` trả anchors, class predictions và bbox predictions. Shape phải được kiểm tra trước training; sai reshape vẫn chạy nhưng ghép nhầm anchor là bug nguy hiểm.

### 14.7.2 Training

Pipeline:

1. match anchors với ground truth;
2. class loss bằng cross-entropy trên anchors;
3. box loss (L1 trong demo) chỉ tính anchors positive nhờ mask;
4. cộng hai loss và update.

Negative anchors áp đảo, nên detector thực tế thường hard-negative mining hoặc focal loss:

$$\operatorname{FL}(p_t)=-\alpha(1-p_t)^\gamma\log p_t.$$

$\gamma$ lớn làm examples đã dễ đóng góp ít hơn.

### 14.7.3 Prediction

Softmax class scores → giải offsets → NMS. Lọc theo confidence threshold và trả rows `[class_id, confidence, xmin, ymin, xmax, ymax]`.

### 14.7.4 Summary

- SSD dự đoán multiscale anchors trong một pass.
- Loss gồm class và offset, offset chỉ có ý nghĩa cho positive anchors.

### 14.7.5 Exercises

1. Smooth L1 dùng vùng bình phương gần 0 và L1 xa 0, gradient ổn định hơn; focal loss tập trung hard examples.
2. Cải tiến: resize vật nhỏ, downsample negatives, cân weight class/box loss và dùng metric detection như mAP thay accuracy đơn thuần.

## 14.8 Region-based CNNs (R-CNNs)

### 14.8.1 R-CNNs

![R-CNN lấy region proposals, chạy CNN trên từng vùng rồi dự đoán class và box](../assets/chapter-14/figure-14-8-1-r-cnn.png "Nguồn: didl.pdf, Figure 14.8.1, trang sách 642, trang PDF 682")

R-CNN: selective search tạo khoảng 2000 region proposals → resize từng vùng → CNN trích features → class/box predictors. Chính xác thời đầu nhưng chậm vì CNN chạy lặp trên nhiều vùng.

### 14.8.2 Fast R-CNN

Fast R-CNN chạy CNN **một lần trên cả ảnh**, map proposals sang feature map, dùng {{term:roi-pooling|RoI pooling}} đưa mọi region về cùng shape, rồi shared head dự đoán class/box. Nó loại phần tính CNN lặp.

### 14.8.3 Faster R-CNN

Faster R-CNN thay selective search bằng Region Proposal Network (RPN) học cùng model. RPN phân loại anchors object/background và tinh chỉnh boxes; proposals đi vào RoI pooling và detection head.

### 14.8.4 Mask R-CNN

Mask R-CNN thêm nhánh FCN dự đoán pixel mask cho từng RoI và dùng RoI Align để tránh lượng tử hóa tọa độ quá thô. Output vừa class/box vừa instance mask.

### 14.8.5 Summary

- R-CNN lặp CNN trên proposals; Fast R-CNN share feature map.
- Faster R-CNN học proposal network.
- Mask R-CNN thêm pixel-level branch cho instance masks.

### 14.8.6 Exercises

1. YOLO cho thấy detection có thể đóng khung thành single regression/set prediction; bài toán khó nằm ở số object thay đổi, assignment và duplicate predictions.
2. SSD one-stage thường nhanh; Faster R-CNN two-stage tạo proposals rồi refine, thường đổi speed lấy accuracy/localization.

<!-- pagebreak -->

## 14.9 Semantic Segmentation and the Dataset

{{term:semantic-segmentation|Semantic segmentation}} dự đoán class từng pixel.

![Ảnh gốc và nhãn semantic ở mức pixel cho dog, cat và background](../assets/chapter-14/figure-14-9-1-semantic-segmentation.png "Nguồn: didl.pdf, Figure 14.9.1, trang sách 648, trang PDF 688")

### 14.9.1 Image Segmentation and Instance Segmentation

- **Image segmentation** không nhất thiết dùng semantic labels; có thể chỉ chia vùng theo tương đồng pixels.
- **Semantic segmentation** biết class nhưng hai con chó cùng class có cùng label.
- {{term:instance-segmentation|Instance segmentation}} còn phân biệt từng object instance.

### 14.9.2 The Pascal VOC2012 Semantic Segmentation Dataset

VOC2012 lưu RGB images và color masks. Sách xây mapping từ RGB color sang class index. Ảnh và mask phải được random crop bằng **cùng tọa độ**. Không resize mask bằng bilinear vì sẽ tạo màu/class IDs không tồn tại; nếu resize label, dùng nearest-neighbor.

### 14.9.3 Summary

- Semantic segmentation chia ảnh thành semantic regions ở pixel level.
- VOC2012 là dataset nền tảng; crop cố định giúp batch và giữ correspondence ảnh-mask.

### 14.9.4 Exercises

1. Ứng dụng: vùng đường/người/xe cho xe tự hành; cơ quan/tổn thương trong y tế; ảnh vệ tinh, nông nghiệp, nền video.
2. Color jitter chỉ áp ảnh, không mask; geometric transforms phải áp đồng bộ. Mix/crop làm mất object cần quy tắc label rõ.

## 14.10 Transposed Convolution

{{term:transposed-convolution|Transposed convolution}} tăng spatial dimensions. Nó không phải phép nghịch đảo đảm bảo khôi phục giá trị input; “transpose” nói về ma trận tuyến tính tương ứng.

### 14.10.1 Basic Operation

![Mỗi phần tử input nhân kernel rồi cộng vào vị trí tương ứng của output lớn hơn](../assets/chapter-14/figure-14-10-1-transposed-convolution.png "Nguồn: didl.pdf, Figure 14.10.1, trang sách 655, trang PDF 695")

Stride 1, no padding: input $n_h\times n_w$, kernel $k_h\times k_w$ cho output:

$$ (n_h+k_h-1)\times(n_w+k_w-1). $$

Regular convolution “gom” một window thành một output; transposed convolution “rải” mỗi input qua kernel rồi cộng phần chồng.

### 14.10.2 Padding, Strides, and Multiple Channels

Trong transposed conv, padding làm **giảm** vùng biên output; stride làm tăng khoảng rải và output. Với channels, layout weight của framework phải được kiểm tra, không suy từ Conv2d bằng tên axis.

### 14.10.3 Connection to Matrix Transposition

Một convolution tuyến tính có thể viết $\mathbf y=\mathbf W\mathbf x$. Transposed convolution với cùng kernel structure tính $\mathbf W^\top\mathbf y$. Shape có thể quay về shape của $x$, nhưng $\mathbf W^\top\mathbf W\mathbf x\neq\mathbf x$ nói chung.

### 14.10.4 Summary

- Regular conv reduce windows; transposed conv broadcast input contributions.
- Matching hyperparameters có thể phục hồi shape, không phục hồi values.
- Tên gọi đến từ transpose của linear operator.

### 14.10.5 Exercises

1. $Z$ và $X$ cùng shape nhưng khác giá trị vì transposed convolution không phải matrix inverse.
2. Ma trận convolution rất lớn và thưa; materialize rồi nhân kém hiệu quả hơn kernel chuyên dụng/im2col tối ưu.

## 14.11 Fully Convolutional Networks

### 14.11.1 The Model

![FCN dùng CNN, 1×1 convolution và transposed convolution để trả output pixel-level](../assets/chapter-14/figure-14-11-1-fcn.png "Nguồn: didl.pdf, Figure 14.11.1, trang sách 660, trang PDF 700")

{{term:fully-convolutional-network|Fully convolutional network}} (FCN): pretrained CNN trích features → 1×1 conv đổi channels thành số classes → transposed conv phóng $H,W$ về kích thước input. Output channel tại $(y,x)$ là logits classes của pixel tương ứng.

### 14.11.2 Initializing Transposed Convolutional Layers

Khởi tạo kernel để transposed conv làm {{term:bilinear-interpolation|bilinear interpolation}} giúp upsampling ban đầu mượt và ổn định hơn random initialization.

### 14.11.3 Reading the Dataset

Dùng VOC loader với crop cố định. Đừng augment validation. Kiểm tra mask IDs nằm trong `[0, num_classes)` (trừ ignore index).

### 14.11.4 Training

Cross-entropy tính theo pixels. Memory tăng theo `batch × classes × H × W`; khi hết VRAM, giảm crop/batch trước khi sửa model tùy tiện.

### 14.11.5 Prediction

Lấy `argmax` theo channel, map class ID về RGB colormap, resize/crop đúng convention. Overlay lên ảnh gốc để kiểm tra biên.

### 14.11.6 Summary

- FCN giữ pipeline hoàn toàn convolutional và upsample về pixel grid.
- Bilinear initialization là điểm khởi đầu tốt cho transposed conv.

### 14.11.7 Exercises

1. Xavier vẫn học nhưng output ban đầu nhiễu hơn và có thể hội tụ chậm.
2. Tune learning rate, crop, augmentation, schedule; giữ validation protocol cố định.
3. Predict mọi pixel bằng `argmax(dim=1)` rồi colorize.
4. Skip connections từ intermediate maps phục hồi chi tiết biên bị mất ở deep coarse feature map.

<!-- pagebreak -->

## 14.12 Neural Style Transfer

{{term:neural-style-transfer|Neural style transfer}} giữ cấu trúc content image nhưng làm statistics features giống style image. Biến được tối ưu là **pixels của ảnh tổng hợp**, không phải CNN weights.

### 14.12.1 Method

![Content image, style image và synthesized image](../assets/chapter-14/figure-14-12-style-transfer.png "Nguồn: didl.pdf, Figure 14.12.1, trang sách 667, trang PDF 707")

![CNN cố định so content/style features; gradient chỉ cập nhật synthesized image](../assets/chapter-14/figure-14-12-2-style-transfer-process.png "Nguồn: didl.pdf, Figure 14.12.2, trang sách 667, trang PDF 707")

Loss tổng:

$$L=\alpha L_{content}+\beta L_{style}+\gamma L_{TV}.$$

- content loss giữ representation ở layer sâu gần content image;
- style loss khớp Gram matrices ở nhiều layers;
- total variation loss giảm nhiễu pixel.

### 14.12.2 Reading the Content and Style Images

Hai ảnh có kích thước khác nhau được đọc riêng. Chọn ảnh bạn có quyền sử dụng; style transfer không xóa vấn đề bản quyền.

### 14.12.3 Preprocessing and Postprocessing

Preprocess resize, tensorize, normalize theo pretrained CNN. Postprocess đảo normalization, clamp `[0,1]`, đổi về ảnh hiển thị. Giữ hai hàm đối xứng giúp tránh màu sai.

### 14.12.4 Extracting Features

Sách dùng VGG-19 pretrained, chọn một số layers làm content/style outputs và freeze weights. Shallow layers thiên texture/cạnh; deeper layers thiên cấu trúc/objects.

### 14.12.5 Defining the Loss Function

Content loss là squared difference features. Style dùng {{term:gram-matrix|Gram matrix}} $G=XX^\top$ (có chuẩn hóa) để đo correlation giữa channels mà ít giữ vị trí tuyệt đối. TV loss phạt khác biệt giữa pixels kề nhau.

### 14.12.6 Initializing the Synthesized Image

Khởi tạo từ content image thường hội tụ nhanh và giữ content hơn khởi tạo noise. Đưa ảnh vào `nn.Parameter` hoặc module chỉ chứa parameter ảnh.

### 14.12.7 Training

Mỗi step forward synthesized image qua frozen CNN, tính ba losses, backprop tới pixels, optimizer update. Sau update có thể clamp hợp lệ và lưu snapshot định kỳ.

### 14.12.8 Summary

- Content/style/TV losses điều khiển ba mục tiêu khác nhau.
- CNN là feature extractor cố định; synthesized image là parameter.
- Gram matrix đại diện style correlations.

### 14.12.9 Exercises

1. Content layer sâu hơn giữ semantic layout nhưng ít chi tiết; style layers khác nhau điều khiển texture ở nhiều scale.
2. Tăng $\alpha$ giữ content; tăng $\beta$ đậm style; tăng $\gamma$ mượt hơn nhưng quá lớn làm mất chi tiết.
3. Đổi cặp ảnh và ghi lại weights/seed để so sánh.
4. Style transfer cho text có thể đổi sentiment/tone nhưng phải giữ meaning; discrete tokens làm optimization khác ảnh.

<!-- pagebreak -->

## 14.13 Image Classification (CIFAR-10) on Kaggle

### 14.13.1 Obtaining and Organizing the Dataset

Competition có 50,000 train images và 300,000 test files (10,000 dùng đánh giá, phần còn lại chống gắn nhãn thủ công), ảnh RGB 32×32, 10 classes. Pipeline tổ chức:

```text
train_valid_test/
  train/class_name/*.png
  valid/class_name/*.png
  train_valid/class_name/*.png
  test/unknown/*.png
```

Validation phải lấy số mẫu cân bằng theo class và không copy cùng ảnh vào train lẫn valid.

### 14.13.2 Image Augmentation

Train: random crop có padding + horizontal flip + normalize. Valid/test: chỉ tensor + normalize. CIFAR nhỏ nên crop/flip có tác dụng rõ.

### 14.13.3 Reading the Dataset

`ImageFolder` suy class từ thư mục. Kiểm tra `class_to_idx` và dùng cùng mapping khi tạo submission.

### 14.13.4 Defining the Model

Sách dùng ResNet-18 điều chỉnh cho 32×32. Không dùng stem stride/pooling quá mạnh như ảnh ImageNet lớn nếu làm mất spatial detail sớm.

### 14.13.5 Defining the Training Function

Training function nhận model, loaders, epochs, learning rate, weight decay, devices và LR scheduler. Các dependencies explicit giúp experiment tái lập.

### 14.13.6 Training and Validating the Model

Tune trên train/valid. Sau khi chốt hyperparameters, train lại với `train_valid` để dùng toàn bộ nhãn; không tiếp tục nhìn test leaderboard để tune vô hạn.

### 14.13.7 Classifying the Testing Set and Submitting Results on Kaggle

Prediction cần đúng thứ tự file IDs. Ghi CSV header `id,label`, map index về class string, rồi kiểm tra vài dòng và số rows.

### 14.13.8 Summary

- Custom image files cần tổ chức đúng format trước khi DataLoader đọc.
- CNN + augmentation tạo baseline competition có hệ thống.

### 14.13.9 Exercises

1. Full dataset với lịch 100 epochs là thí nghiệm tốn compute; log seed, split, config và best valid accuracy trước khi nộp.
2. Bỏ augmentation thường làm train cao hơn nhưng valid thấp hơn; đo chênh lệch qua nhiều runs.

## 14.14 Dog Breed Identification (ImageNet Dogs) on Kaggle

### 14.14.1 Obtaining and Organizing the Dataset

Dataset là subset ImageNet: 10,222 train, 10,357 test JPEG, 120 breeds, kích thước ảnh thay đổi. Tổ chức thư mục tương tự CIFAR nhưng split validation tối thiểu theo từng breed.

### 14.14.2 Image Augmentation

Ảnh lớn: train random resized crop 224, flip, color jitter; test resize + center crop. Normalize đúng mean/std ImageNet vì dùng pretrained model.

### 14.14.3 Reading the Dataset

Batch image đã thống nhất shape `(batch,3,224,224)`. Với 120 classes, kiểm tra class ordering giữa loader và submission columns.

### 14.14.4 Fine-Tuning a Pretrained Model

Sách dùng ResNet-34 pretrained làm features, sau đó small output network có linear/ReLU/dropout/linear 120 classes. Freeze feature extractor giảm compute/memory; unfreeze dần có thể cải thiện khi có tài nguyên.

### 14.14.5 Defining the Training Function

Hàm train dùng cross-entropy, Adam, weight decay và scheduler. Không để augmentation của validation lọt vào vì metric sẽ dao động theo random crop.

### 14.14.6 Training and Validating the Model

Tune với validation split trước; sau đó fit model cuối trên train+valid. Theo dõi top-1 accuracy và loss, nhưng cân nhắc top-k vì 120 breeds rất giống nhau.

### 14.14.7 Classifying the Testing Set and Submitting Results on Kaggle

Competition yêu cầu xác suất cho 120 breeds. Apply softmax, tạo một column mỗi class đúng thứ tự, giữ `id`, và xác nhận mỗi row có xác suất tổng xấp xỉ 1.

### 14.14.8 Summary

- ImageNet Dogs lớn/biến thiên kích thước hơn CIFAR nên augmentation khác.
- Pretrained ImageNet features + head nhỏ tiết kiệm compute/memory.

### 14.14.9 Exercises

1. Tăng batch/epochs chỉ hữu ích nếu scheduler và validation cho thấy model còn học; batch lớn có thể cần scale learning rate.
2. Backbone sâu hơn không luôn tốt hơn với budget cố định. Tune crop, frozen layers, discriminative LR, weight decay và augmentation có kiểm soát.

## Điểm hay và ý nghĩa

Chương 14 nối ba tầng hiểu ảnh thành một hệ thống: **invariance** (augmentation), **localization** (boxes/anchors/proposals) và **dense prediction** (segmentation). Transposed convolution/FCN cho thấy output shape không chỉ là chi tiết code mà là phần của định nghĩa bài toán. Style transfer lại đảo vai trò quen thuộc: model đứng yên, input image được tối ưu.

## Sau chương này bạn làm được gì?

- Chọn transform không phá nhãn và tách train/test preprocessing.
- Debug box format/IoU/NMS bằng hình vẽ và shape.
- So sánh one-stage với two-stage detector.
- Tạo synchronized image-mask pipeline cho segmentation.
- Đọc output `(N,C,H,W)` thành pixel classes.
- Thiết kế experiment/submission không leakage.

## Tóm tắt kiến thức

1. Augmentation dạy invariance; fine-tuning tái sử dụng representation.
2. Detection = anchors/proposals + class + box regression + duplicate removal.
3. Multiscale features xử lý vật nhiều kích thước.
4. Segmentation cần alignment ở pixel level.
5. Transposed conv tăng resolution; FCN biến features thành dense labels.
6. Style transfer tối ưu pixels bằng losses trong feature space.

## Thuật ngữ cần nhớ

| English term | Hiểu ngắn gọn |
|---|---|
| Bounding box / anchor box | box thật dự đoán / box ứng viên ban đầu |
| IoU / NMS | đo overlap / giảm predictions trùng |
| SSD | one-stage multiscale detector |
| RPN / RoI pooling | học proposals / đưa region về feature shape cố định |
| Semantic / instance segmentation | class từng pixel / tách từng vật |
| Transposed convolution | linear transpose operation để upsample |
| FCN | CNN trả dense pixel-level outputs |
| Gram matrix | channel correlations dùng đại diện style |

## Nguồn và phạm vi

- Nguồn chính: *Dive into Deep Learning*, Chương 14, trang sách 592–689, trang PDF vật lý 632–729.
- Toàn bộ 14 mục, summaries và exercises đã được đọc/đối chiếu theo đúng thứ tự PDF.
- Figures 14.7.1, 14.8.1, 14.9.1, 14.10.1, 14.11.1, 14.12.1 và 14.12.2 được trích trực tiếp từ `didl.pdf`; code rút gọn và các lưu ý convention/debug là giảng giải bổ sung của vở.
