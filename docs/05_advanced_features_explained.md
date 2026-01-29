# 📘 شرح شامل لـ Advanced Features في Pose Format

## 🎯 نظرة عامة

هذا الدليل يشرح بالتفصيل جميع الميزات المتقدمة في مكتبة `pose_format` والتي تستخدم في معالجة وتحليل بيانات الحركة والإشارات. هذه الأدوات ضرورية لأي مشروع يتعامل مع:
- تدريب نماذج التعلم الآلي على بيانات الحركة
- تحليل لغة الإشارة
- تطبيقات التعرف على الحركة والإيماءات
- معالجة فيديوهات التمارين الرياضية

---

## 📚 جدول المحتويات

1. [الإعدادات الأولية](#1-الإعدادات-الأولية)
2. [التطبيع (Normalization)](#2-التطبيع-normalization)
3. [تضخيم البيانات (Data Augmentation)](#3-تضخيم-البيانات-data-augmentation)
4. [الاستيفاء وتغيير FPS](#4-الاستيفاء-وتغيير-fps)
5. [قص وتحديد المكونات](#5-قص-وتحديد-المكونات)
6. [حساب Bounding Box](#6-حساب-bounding-box)
7. [تسقيط الإطارات (Frame Dropout)](#7-تسقيط-الإطارات-frame-dropout)
8. [قلب الـ Pose (Flip)](#8-قلب-الـ-pose-flip)
9. [التحويل بين Backends](#9-التحويل-بين-backends)
10. [دوال مساعدة إضافية](#10-دوال-مساعدة-إضافية)
11. [خط معالجة شامل](#11-خط-معالجة-شامل)

---

## 1. الإعدادات الأولية

### 📦 استيراد المكتبات

```python
import numpy as np
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
```

**الشرح:**
- `numpy`: للتعامل مع المصفوفات والعمليات الرياضية
- `Pose`: الكلاس الرئيسي للتعامل مع بيانات الحركة
- `PoseVisualizer`: لتصوير وعرض الحركات

### 🛠️ الدوال المساعدة الأساسية

```python
def load_pose(path: str) -> Pose:
    with open(path, 'rb') as f:
        return Pose.read(f.read())

def save_as_pose(pose: Pose, output_path: str):
    with open(output_path, 'wb') as f:
        pose.write(f)
    print(f"✅ Saved to: {output_path}")
```

**الشرح:**
- `load_pose()`: تحميل ملف pose من القرص
- `save_as_pose()`: حفظ كائن Pose إلى ملف
- الملفات تُقرأ وتُكتب بصيغة binary (`'rb'` و `'wb'`)

**مثال عملي:**
```python
# تحميل فيديو إشارة
pose = load_pose('../data/pose_files/example.pose')

# حفظ نسخة معدلة
save_as_pose(pose, '../output/modified_pose.pose')
```

---

## 2. التطبيع (Normalization)

### 🎯 ما هو التطبيع ولماذا نحتاجه؟

**المشكلة:**
عند تسجيل فيديوهات مختلفة، قد يكون:
- الشخص قريب أو بعيد عن الكاميرا
- الشخص في جانب الصورة أو في المنتصف
- أشخاص بأحجام مختلفة (طويل/قصير، طفل/بالغ)

**الحل:**
التطبيع يجعل جميع البيانات في مقياس موحد، مما يسهل المقارنة والتدريب.

---

### 2.1 التطبيع الأساسي (Basic Normalization)

```python
def normalize_pose_basic(pose: Pose) -> Pose:
    pose_copy = pose.copy()
    pose_copy.normalize()
    return pose_copy
```

**كيف يعمل:**
1. **المركز (Center)**: ينقل نقطة المنتصف بين الكتفين إلى الإحداثي (0, 0)
2. **المقياس (Scale)**: يجعل المسافة بين الكتفين = 1

**مثال واقعي:**

```python
# لديك فيديوهات لإشارة "مرحبا" من أشخاص مختلفين
pose_person1 = load_pose('person1_hello.pose')  # شخص طويل، بعيد
pose_person2 = load_pose('person2_hello.pose')  # شخص قصير، قريب

# بدون تطبيع: القيم مختلفة جداً
# بعد التطبيع: القيم متشابهة ويمكن مقارنتها
normalized1 = normalize_pose_basic(pose_person1)
normalized2 = normalize_pose_basic(pose_person2)

# الآن يمكن للنموذج التعرف على أن كلاهما نفس الإشارة
```

**سيناريو استخدام:**
- تدريب نموذج تعرف على الإشارات من مصادر مختلفة
- مقارنة أداء حركات رياضية من رياضيين مختلفين
- توحيد بيانات الرقص من فيديوهات مختلفة

---

### 2.2 التطبيع المخصص (Custom Normalization)

```python
def normalize_pose_custom(pose: Pose, 
                          component1: str, point1: str,
                          component2: str, point2: str,
                          scale_factor: float = 1.0) -> Pose:
    pose_copy = pose.copy()
    norm_info = pose_copy.header.normalization_info(
        p1=(component1, point1),
        p2=(component2, point2)
    )
    pose_copy.normalize(norm_info, scale_factor=scale_factor)
    return pose_copy
```

**الشرح:**
يسمح لك باختيار **أي نقطتين** للتطبيع بدلاً من الكتفين الافتراضيين.

**أمثلة عملية:**

#### مثال 1: تحليل إشارات اليد

```python
# التطبيع بناءً على الرسغين (مهم لإشارات اليد)
normalized = normalize_pose_custom(
    pose,
    "POSE_LANDMARKS", "RIGHT_WRIST",
    "POSE_LANDMARKS", "LEFT_WRIST"
)
```

**متى تستخدم:**
- تحليل لغة الإشارة (حركة اليدين أهم من الجسم)
- التعرف على الإيماءات اليدوية
- تطبيقات التحكم بالإيماءات

#### مثال 2: تحليل حركات الساقين

```python
# التطبيع بناءً على الوركين (مهم لتحليل المشي/الجري)
normalized = normalize_pose_custom(
    pose,
    "POSE_LANDMARKS", "RIGHT_HIP",
    "POSE_LANDMARKS", "LEFT_HIP"
)
```

**متى تستخدم:**
- تحليل أنماط المشي
- تقييم حركات اللاعبين الرياضيين (الركض، القفز)
- تطبيقات إعادة التأهيل الطبي

#### مثال 3: تحليل تعبيرات الوجه

```python
# التطبيع بناءً على العينين
normalized = normalize_pose_custom(
    pose,
    "FACE_LANDMARKS", "LEFT_EYE",
    "FACE_LANDMARKS", "RIGHT_EYE",
    scale_factor=2.0  # توسيع المقياس
)
```

---

### 2.3 التطبيع الإحصائي (Distribution Normalization)

```python
def normalize_distribution(pose: Pose, axis: tuple = (0, 1)) -> tuple:
    pose_copy = pose.copy()
    mu, std = pose_copy.normalize_distribution(axis=axis)
    return pose_copy, mu, std
```

**كيف يعمل:**
يطبق الصيغة الإحصائية الشهيرة:

$$X_{normalized} = \frac{X - \mu}{\sigma}$$

حيث:
- $\mu$ = المتوسط (Mean)
- $\sigma$ = الانحراف المعياري (Standard Deviation)

**النتيجة:**
- المتوسط = 0
- الانحراف المعياري = 1

**معاملات axis:**

```python
# axis=(0, 1) - عبر الإطارات والأشخاص
# يحسب المتوسط والانحراف لكل نقطة على حدة
normalized, mu, std = normalize_distribution(pose, axis=(0, 1))

# axis=(0, 1, 2) - عبر كل شيء
# يحسب متوسط وانحراف واحد لجميع النقاط
normalized, mu, std = normalize_distribution(pose, axis=(0, 1, 2))
```

**مثال واقعي - نموذج تعلم آلي:**

```python
# التدريب
train_poses = []
for video in training_videos:
    pose = load_pose(video)
    normalized, mu, std = normalize_distribution(pose)
    train_poses.append(normalized)
    
# حفظ معاملات التطبيع للاستخدام لاحقاً
np.save('mu.npy', mu)
np.save('std.npy', std)

# الاستدلال (Testing)
test_pose = load_pose('new_video.pose')
mu = np.load('mu.npy')
std = np.load('std.npy')

# استخدام نفس المعاملات
test_pose_data = (test_pose.body.data - mu) / std
```

### 2.4 إلغاء التطبيع (Unnormalization)

```python
def unnormalize_distribution(pose: Pose, mu, std) -> Pose:
    pose_copy = pose.copy()
    pose_copy.unnormalize_distribution(mu, std)
    return pose_copy
```

**لماذا نحتاجه:**
بعد معالجة البيانات أو التنبؤ، قد نريد إرجاع القيم الأصلية للعرض أو التحليل.

**مثال - توليد حركات جديدة:**

```python
# 1. تطبيع البيانات
normalized, mu, std = normalize_distribution(original_pose)

# 2. تدريب نموذج توليدي (GAN, VAE)
model.train(normalized)

# 3. توليد حركة جديدة
generated_normalized = model.generate()

# 4. إرجاع القيم الأصلية للعرض
generated_original = unnormalize_distribution(generated_normalized, mu, std)

# 5. عرض الحركة الناتجة
visualize(generated_original)
```

---

## 3. تضخيم البيانات (Data Augmentation)

### 🎯 ما هو تضخيم البيانات؟

**المشكلة:**
- لديك 100 فيديو فقط لتدريب نموذج
- النموذج يحتاج آلاف الأمثلة ليتعلم بشكل جيد
- جمع المزيد من البيانات مكلف ويأخذ وقت

**الحل:**
تضخيم البيانات = إنشاء نسخ معدلة قليلاً من البيانات الموجودة

---

### 3.1 التضخيم ثنائي الأبعاد (2D Augmentation)

```python
def augment_pose_2d(pose: Pose, 
                    rotation_std: float = 0.2,
                    shear_std: float = 0.2,
                    scale_std: float = 0.2) -> Pose:
    pose_copy = pose.copy()
    augmented = pose_copy.augment2d(
        rotation_std=rotation_std,
        shear_std=shear_std,
        scale_std=scale_std
    )
    return augmented
```

**التحويلات التي تحدث:**

1. **Rotation (الدوران)**: دوران عشوائي خفيف
2. **Shear (القص)**: إمالة خفيفة
3. **Scale (التحجيم)**: تكبير أو تصغير خفيف

**تصور التحويلات:**

```
الأصل:        بعد Rotation:    بعد Shear:      بعد Scale:
  |              /              /|             |
  |             /              / |            ||
  |            /              /  |            ||
```

**أمثلة عملية:**

#### مثال 1: تدريب نموذج تعرف على الإشارات

```python
# لديك 50 فيديو لإشارة "شكراً"
original_pose = load_pose('thank_you.pose')

# إنشاء 10 نسخ مختلفة قليلاً
augmented_dataset = []
for i in range(10):
    aug = augment_pose_2d(
        original_pose,
        rotation_std=0.15,  # دوران بسيط
        shear_std=0.1,      # إمالة بسيطة
        scale_std=0.15      # تكبير/تصغير بسيط
    )
    augmented_dataset.append(aug)

# الآن لديك 50 × 10 = 500 مثال للتدريب!
```

#### مثال 2: جعل النموذج أكثر قوة

```python
# بدون تضخيم: النموذج يتعلم فقط زاوية تصوير واحدة
# مع التضخيم: النموذج يتعلم التعرف على الإشارة من زوايا مختلفة

def create_robust_dataset(pose: Pose, num_augmentations: int = 10):
    dataset = [pose]  # الأصل
    
    for i in range(num_augmentations):
        # تغيير عشوائي في كل مرة
        rotation = np.random.uniform(0.05, 0.3)
        shear = np.random.uniform(0.05, 0.2)
        scale = np.random.uniform(0.05, 0.2)
        
        aug = pose.copy().augment2d(
            rotation_std=rotation,
            shear_std=shear,
            scale_std=scale
        )
        dataset.append(aug)
    
    return dataset
```

**متى تستخدم التضخيم:**

✅ **استخدم عندما:**
- لديك بيانات محدودة
- تريد نموذج أكثر قوة (robust)
- تريد تجنب Overfitting

❌ **لا تستخدم عندما:**
- التحويلات تغير معنى البيانات (مثل: قلب إشارة "يمين" لتصبح "يسار")
- لديك بيانات كافية بالفعل

---

### 3.2 إنشاء مجموعة بيانات مضخمة

```python
def create_augmented_dataset(pose: Pose, num_augmentations: int = 10) -> list:
    augmented_poses = [pose.copy()]  # الأصل
    
    for i in range(num_augmentations):
        rotation = np.random.uniform(0.05, 0.3)
        shear = np.random.uniform(0.05, 0.2)
        scale = np.random.uniform(0.05, 0.2)
        
        aug = pose.copy().augment2d(
            rotation_std=rotation,
            shear_std=shear,
            scale_std=scale
        )
        augmented_poses.append(aug)
    
    return augmented_poses
```

**سيناريو كامل - تدريب نموذج:**

```python
# 1. تحميل جميع الفيديوهات الأصلية
original_videos = [
    'sign1.pose', 'sign2.pose', 'sign3.pose'
]

# 2. تضخيم كل فيديو
all_data = []
for video_path in original_videos:
    pose = load_pose(video_path)
    augmented = create_augmented_dataset(pose, num_augmentations=20)
    all_data.extend(augmented)

print(f"البيانات الأصلية: {len(original_videos)} فيديو")
print(f"بعد التضخيم: {len(all_data)} فيديو")
# النتيجة: 3 → 63 فيديو (3 × 21)

# 3. استخدامها في التدريب
X_train = [pose.body.data for pose in all_data]
model.fit(X_train, y_train)
```

---

## 4. الاستيفاء وتغيير FPS

### 🎯 ما هي مشكلة FPS المختلف؟

**المشكلة:**
```
فيديو 1: 30 FPS (30 إطار في الثانية)
فيديو 2: 24 FPS
فيديو 3: 60 FPS
فيديو 4: 25 FPS
```

لتدريب نموذج، نحتاج **FPS موحد** لجميع الفيديوهات.

---

### 4.1 تغيير معدل الإطارات

```python
def change_fps(pose: Pose, new_fps: float, kind: str = 'cubic') -> Pose:
    pose_copy = pose.copy()
    original_fps = pose_copy.body.fps
    interpolated = pose_copy.interpolate(new_fps=new_fps, kind=kind)
    return interpolated
```

**أنواع الاستيفاء (Interpolation Methods):**

1. **'linear'**: خط مستقيم بين النقاط
   - سريع
   - حركة قد تبدو متقطعة

2. **'quadratic'**: منحنى من الدرجة الثانية
   - متوسط السرعة
   - حركة أنعم

3. **'cubic'**: منحنى من الدرجة الثالثة
   - أبطأ قليلاً
   - حركة ناعمة جداً (الأفضل للحركات الطبيعية)

**تصور الاستيفاء:**

```
الإطارات الأصلية:  ●-----●-----●-----●
                    1     5     9    13

linear:            ●-○-○-●-○-○-●-○-○-●
cubic (أنعم):      ●~○~○~●~○~○~●~○~○~●
```

**أمثلة عملية:**

#### مثال 1: توحيد FPS للتدريب

```python
# لديك فيديوهات بـ FPS مختلف
videos = [
    ('video1.pose', 30),
    ('video2.pose', 24),
    ('video3.pose', 60),
]

# توحيد الكل إلى 25 FPS
unified_fps = 25
unified_videos = []

for video_path, original_fps in videos:
    pose = load_pose(video_path)
    unified = change_fps(pose, new_fps=unified_fps, kind='cubic')
    unified_videos.append(unified)

# الآن جميع الفيديوهات بنفس FPS
```

#### مثال 2: Downsampling (تقليل الإطارات)

```python
# فيديو بـ 60 FPS (تفاصيل كثيرة، حجم كبير)
high_fps_pose = load_pose('high_quality.pose')

# تقليل إلى 30 FPS (تقليل الحجم، سرعة معالجة أعلى)
reduced = change_fps(high_fps_pose, new_fps=30)

print(f"الحجم الأصلي: {high_fps_pose.body.data.shape[0]} إطار")
print(f"بعد التقليل: {reduced.body.data.shape[0]} إطار")
# مثال: 600 → 300 إطار (50% أقل!)
```

#### مثال 3: Upsampling (زيادة الإطارات)

```python
# فيديو قديم بـ 15 FPS (حركة متقطعة)
old_video = load_pose('old_recording.pose')

# زيادة إلى 30 FPS (حركة أنعم)
smooth = change_fps(old_video, new_fps=30, kind='cubic')

# الآن الحركة تبدو أنعم وأكثر طبيعية
```

---

### 4.2 ملء الإطارات المفقودة

```python
def interpolate_missing_frames(pose: Pose) -> Pose:
    pose_copy = pose.copy()
    interpolated = pose_copy.interpolate(new_fps=None, kind='linear')
    return interpolated
```

**متى تحدث إطارات مفقودة؟**

```python
# مثال: في بعض الإطارات، لم يتم اكتشاف اليد
frame 1: [x=0.5, y=0.3]  ✓ تم اكتشاف اليد
frame 2: [x=NaN, y=NaN]  ✗ لم يتم اكتشاف اليد (خلف الجسم)
frame 3: [x=NaN, y=NaN]  ✗ لم يتم اكتشاف اليد
frame 4: [x=0.7, y=0.4]  ✓ تم اكتشاف اليد مجدداً
```

**الحل بالاستيفاء:**

```python
# ملء القيم المفقودة بالاستيفاء
pose_filled = interpolate_missing_frames(pose)

# النتيجة:
frame 1: [x=0.5, y=0.3]  ✓ أصلي
frame 2: [x=0.57, y=0.33] ✓ محسوب بالاستيفاء
frame 3: [x=0.63, y=0.37] ✓ محسوب بالاستيفاء
frame 4: [x=0.7, y=0.4]  ✓ أصلي
```

**مثال واقعي - تحسين جودة البيانات:**

```python
# 1. تحميل بيانات فيها قيم مفقودة
raw_pose = load_pose('noisy_video.pose')

# 2. ملء الإطارات المفقودة
clean_pose = interpolate_missing_frames(raw_pose)

# 3. حفظ النسخة المحسّنة
save_as_pose(clean_pose, 'cleaned_video.pose')

# الآن البيانات جاهزة للتدريب بدون قيم مفقودة
```

---

## 5. قص وتحديد المكونات

### 🎯 ما هي المكونات (Components)؟

**المكونات الرئيسية في MediaPipe:**

```python
POSE_LANDMARKS         # 33 نقطة للجسم
FACE_LANDMARKS         # 468 نقطة للوجه
LEFT_HAND_LANDMARKS    # 21 نقطة لليد اليسرى
RIGHT_HAND_LANDMARKS   # 21 نقطة لليد اليمنى
POSE_WORLD_LANDMARKS   # نسخة 3D من نقاط الجسم
```

**المجموع:** 33 + 468 + 21 + 21 + 33 = **576 نقطة!**

**المشكلة:**
- معظم التطبيقات لا تحتاج كل هذه النقاط
- حجم البيانات كبير جداً
- وقت المعالجة طويل

---

### 5.1 استخراج مكونات محددة

```python
def get_specific_components(pose: Pose, components: list) -> Pose:
    filtered = pose.get_components(components)
    return filtered
```

**أمثلة عملية:**

#### مثال 1: لغة الإشارة - اليدين فقط

```python
# لغة الإشارة تعتمد بشكل أساسي على اليدين
hands_only = get_specific_components(pose, [
    'LEFT_HAND_LANDMARKS',
    'RIGHT_HAND_LANDMARKS'
])

print(f"النقاط الأصلية: {pose.body.data.shape[2]}")     # 576
print(f"بعد التصفية: {hands_only.body.data.shape[2]}") # 42 فقط!

# تقليل 93% من البيانات!
# سرعة معالجة أعلى بكثير
```

#### مثال 2: تحليل تعبيرات الوجه

```python
# تطبيق يحلل مشاعر الشخص من وجهه
face_only = get_specific_components(pose, [
    'FACE_LANDMARKS'
])

# الآن لديك فقط 468 نقطة للوجه بدلاً من 576
# دقة أعلى في تحليل الوجه
```

#### مثال 3: تحليل حركة الجسم

```python
# تطبيق رياضي لتحليل وضعية الجسم
body_only = get_specific_components(pose, [
    'POSE_LANDMARKS'
])

# 33 نقطة فقط - كافية لتحليل الوضعية
# مفيد لتطبيقات اللياقة، اليوغا، إلخ
```

#### مثال 4: جسم + يدين (الأكثر شيوعاً)

```python
# معظم تطبيقات لغة الإشارة
body_and_hands = get_specific_components(pose, [
    'POSE_LANDMARKS',
    'LEFT_HAND_LANDMARKS',
    'RIGHT_HAND_LANDMARKS'
])

# 33 + 21 + 21 = 75 نقطة
# توازن مثالي بين التفاصيل والحجم
```

---

### 5.2 استخراج نقاط محددة من المكونات

```python
def get_specific_points(pose: Pose, components: list, points_dict: dict) -> Pose:
    filtered = pose.get_components(components, points=points_dict)
    return filtered
```

**مثال - الجزء العلوي من الجسم فقط:**

```python
upper_body = get_specific_points(
    pose,
    ['POSE_LANDMARKS'],
    {
        'POSE_LANDMARKS': [
            'NOSE',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP'
        ]
    }
)

# فقط 9 نقاط من 33!
# مثالي لتحليل إشارات اليد التي لا تحتاج للساقين
```

**مثال - تحليل المشي (الساقين فقط):**

```python
legs_only = get_specific_points(
    pose,
    ['POSE_LANDMARKS'],
    {
        'POSE_LANDMARKS': [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
    }
)

# مثالي لتطبيقات تحليل المشي والجري
```

---

### 5.3 إزالة مكونات

```python
def remove_components(pose: Pose, components_to_remove: list) -> Pose:
    filtered = pose.remove_components(components_to_remove)
    return filtered
```

**مثال - إزالة POSE_WORLD_LANDMARKS:**

```python
# POSE_WORLD_LANDMARKS تكون كبيرة الحجم وأحياناً غير مفيدة
pose_no_world = remove_components(pose, ['POSE_WORLD_LANDMARKS'])

# تقليل الحجم بدون فقدان معلومات مهمة
```

**مثال - إزالة الوجه للخصوصية:**

```python
# تطبيق يحتاج حركة الجسم فقط بدون تعريف الشخص
no_face = remove_components(pose, ['FACE_LANDMARKS'])

# حماية الخصوصية - لا يمكن التعرف على الوجه
```

---

## 6. حساب Bounding Box

### 🎯 ما هو Bounding Box؟

**Bounding Box** = المستطيل الذي يحيط بمجموعة من النقاط

```
    TOP_LEFT ●─────────────┐
             │             │
             │   النقاط    │
             │      ●  ● ● │
             │     ● ●  ●  │
             └─────────────● BOTTOM_RIGHT
```

---

### 6.1 حساب Bounding Box لكل المكونات

```python
def compute_bounding_box(pose: Pose) -> Pose:
    bbox_pose = pose.bbox()
    return bbox_pose
```

**النتيجة:**
كل مكون يتحول إلى نقطتين فقط:
- TOP_LEFT (أعلى يسار)
- BOTTOM_RIGHT (أسفل يمين)

**مثال:**

```python
# الأصل: 21 نقطة لليد اليمنى
original = load_pose('sign.pose')
print(original.body.data.shape)  # [frames, people, 21, dimensions]

# Bounding Box: نقطتان فقط
bbox = compute_bounding_box(original)
print(bbox.body.data.shape)  # [frames, people, 2, dimensions]

# تقليل البيانات من 21 → 2 نقطة!
```

---

### 6.2 استخراج Bounding Box لليدين

```python
def get_hands_bounding_box(pose: Pose) -> dict:
    # 1. استخراج اليدين فقط
    hands = pose.get_components([
        'LEFT_HAND_LANDMARKS',
        'RIGHT_HAND_LANDMARKS'
    ])
    
    # 2. حساب Bounding Box
    bbox = hands.bbox()
    
    # 3. استخراج الإحداثيات
    data = bbox.body.data
    result = {}
    
    for frame_idx in range(data.shape[0]):
        left_tl = data[frame_idx, 0, 0, :2]   # اليد اليسرى - أعلى يسار
        left_br = data[frame_idx, 0, 1, :2]   # اليد اليسرى - أسفل يمين
        right_tl = data[frame_idx, 0, 2, :2]  # اليد اليمنى - أعلى يسار
        right_br = data[frame_idx, 0, 3, :2]  # اليد اليمنى - أسفل يمين
        
        result[frame_idx] = {
            'left_hand': {'top_left': left_tl, 'bottom_right': left_br},
            'right_hand': {'top_left': right_tl, 'bottom_right': right_br}
        }
    
    return result
```

**استخدامات عملية:**

#### استخدام 1: تتبع موقع اليد

```python
boxes = get_hands_bounding_box(pose)

# الإطار 10
frame_10 = boxes[10]
print(f"اليد اليسرى في: {frame_10['left_hand']['top_left']}")
print(f"اليد اليمنى في: {frame_10['right_hand']['top_left']}")

# مفيد لمعرفة أين توجد اليدين في كل إطار
```

#### استخدام 2: اكتشاف تقاطع اليدين

```python
def hands_are_crossing(bbox_frame):
    """هل اليدان متقاطعتان؟"""
    left = bbox_frame['left_hand']
    right = bbox_frame['right_hand']
    
    # فحص التقاطع
    if (left['bottom_right'][0] > right['top_left'][0] and
        left['top_left'][0] < right['bottom_right'][0]):
        return True
    return False

# فحص كل الإطارات
boxes = get_hands_bounding_box(pose)
for frame_idx, bbox in boxes.items():
    if hands_are_crossing(bbox):
        print(f"الإطار {frame_idx}: اليدان متقاطعتان!")
        
# مفيد لإشارات مثل "X" أو "صلاة"
```

#### استخدام 3: حساب حجم اليد

```python
def calculate_hand_size(bbox):
    """حساب حجم اليد (عرض × ارتفاع)"""
    width = bbox['bottom_right'][0] - bbox['top_left'][0]
    height = bbox['bottom_right'][1] - bbox['top_left'][1]
    area = width * height
    return area

boxes = get_hands_bounding_box(pose)
left_sizes = []

for frame_idx, bbox in boxes.items():
    size = calculate_hand_size(bbox['left_hand'])
    left_sizes.append(size)

# تحليل: هل اليد تقترب أو تبتعد عن الكاميرا؟
print(f"متوسط حجم اليد: {np.mean(left_sizes)}")
print(f"أكبر حجم: {np.max(left_sizes)} - أصغر حجم: {np.min(left_sizes)}")
```

---

## 7. تسقيط الإطارات (Frame Dropout)

### 🎯 لماذا نسقط إطارات؟

**الهدف:** جعل النموذج أكثر قوة (robust) عند التعامل مع:
- فيديوهات بسرعات مختلفة
- إطارات مفقودة
- جودة منخفضة

**الفكرة:**
- إذا تدرب النموذج على بيانات كاملة فقط → قد يفشل مع بيانات ناقصة
- إذا تدرب على بيانات ناقصة أحياناً → يصبح أقوى وأكثر مرونة

---

### 7.1 التسقيط المنتظم (Uniform Dropout)

```python
def frame_dropout_uniform(pose: Pose, 
                          dropout_min: float = 0.2,
                          dropout_max: float = 1.0) -> tuple:
    dropped_pose, selected_indices = pose.frame_dropout_uniform(
        dropout_min=dropout_min,
        dropout_max=dropout_max
    )
    return dropped_pose, selected_indices
```

**كيف يعمل:**
```python
# مثال: dropout_min=0.5, dropout_max=0.8

# الأصل: 100 إطار
# النتيجة: عدد عشوائي بين 50-80 إطار

# تشغيل 1: 65 إطار (65%)
# تشغيل 2: 72 إطار (72%)
# تشغيل 3: 58 إطار (58%)
```

**مثال عملي - تدريب نموذج robust:**

```python
# تحميل البيانات
original_pose = load_pose('training_video.pose')

# إنشاء نسخ بإطارات مختلفة
training_variations = []

for i in range(10):
    # في كل مرة، عدد إطارات مختلف
    dropped, indices = frame_dropout_uniform(
        original_pose,
        dropout_min=0.6,  # على الأقل 60%
        dropout_max=0.9   # على الأكثر 90%
    )
    training_variations.append(dropped)

# الآن النموذج يتعلم من نفس الحركة بسرعات مختلفة
```

---

### 7.2 التسقيط الطبيعي (Normal Dropout)

```python
def frame_dropout_normal(pose: Pose,
                         dropout_mean: float = 0.5,
                         dropout_std: float = 0.1) -> tuple:
    dropped_pose, selected_indices = pose.frame_dropout_normal(
        dropout_mean=dropout_mean,
        dropout_std=dropout_std
    )
    return dropped_pose, selected_indices
```

**الفرق عن Uniform:**

```python
Uniform:  نسبة عشوائية بين حدين
          │═══════════│

Normal:   توزيع طبيعي حول متوسط
              ╱█╲
            ╱█████╲
          ╱█████████╲
```

**متى تستخدم كل نوع؟**

| الحالة | استخدم |
|--------|---------|
| تريد تنوع كبير | Uniform |
| تريد قيم قريبة من رقم معين | Normal |
| لا تهتم بالتوزيع | Uniform أبسط |

---

### 7.3 سيناريو كامل - Data Augmentation بالتسقيط

```python
def create_speed_varied_dataset(pose: Pose, num_variations: int = 5):
    """
    إنشاء نسخ من نفس الإشارة بسرعات مختلفة
    """
    dataset = []
    
    for i in range(num_variations):
        # سرعة بطيئة (إطارات كثيرة)
        if i < num_variations // 2:
            dropped, _ = frame_dropout_uniform(
                pose,
                dropout_min=0.8,  # إبقاء 80-95%
                dropout_max=0.95
            )
        # سرعة سريعة (إطارات قليلة)
        else:
            dropped, _ = frame_dropout_uniform(
                pose,
                dropout_min=0.4,  # إبقاء 40-60%
                dropout_max=0.6
            )
        
        dataset.append(dropped)
    
    return dataset

# النموذج الآن يتعرف على الإشارة سواء كانت بطيئة أو سريعة!
```

---

## 8. قلب الـ Pose (Flip)

### 🎯 ما هو القلب ولماذا نستخدمه؟

**القلب (Flip)** = انعكاس البيانات على محور معين

```
الأصل:    قلب أفقي (X):    قلب رأسي (Y):
  ●              ●                 ●
 /│\            \│/               \│/
  │              │                 │
 / \            \ /               ┴ ┴
```

---

### 8.1 قلب على المحاور المختلفة

```python
def flip_pose(pose: Pose, axis: int = 0) -> Pose:
    flipped = pose.flip(axis=axis)
    return flipped
```

**المحاور:**
- `axis=0`: قلب أفقي (X) - مثل المرآة
- `axis=1`: قلب رأسي (Y) - رأساً على عقب
- `axis=2`: قلب العمق (Z) - الأمام/الخلف

**أمثلة عملية:**

#### مثال 1: مضاعفة بيانات التدريب

```python
# لديك 100 فيديو لإشارة "مرحبا" باليد اليمنى
right_hand_sign = load_pose('hello_right.pose')

# إنشاء نسخة باليد اليسرى (قلب أفقي)
left_hand_sign = flip_pose(right_hand_sign, axis=0)

# الآن لديك 200 مثال بدلاً من 100!
# النموذج يتعرف على الإشارة باليد اليمنى أو اليسرى
```

#### مثال 2: تصحيح اتجاه الفيديو

```python
# فيديو تم تصويره بالمقلوب
upside_down_video = load_pose('wrong_orientation.pose')

# تصحيح الاتجاه
corrected = flip_pose(upside_down_video, axis=1)

# الآن الفيديو بالاتجاه الصحيح
```

---

### 8.2 القلب كتضخيم للبيانات

```python
def mirror_pose_for_augmentation(pose: Pose) -> Pose:
    """قلب أفقي للتضخيم"""
    mirrored = pose.flip(axis=0)
    return mirrored
```

**سيناريو كامل - تدريب نموذج متماثل:**

```python
# مجموعة بيانات أصلية
original_dataset = [
    load_pose('sign1.pose'),
    load_pose('sign2.pose'),
    load_pose('sign3.pose'),
]

# إضافة النسخ المقلوبة
augmented_dataset = []

for pose in original_dataset:
    augmented_dataset.append(pose)           # الأصل
    augmented_dataset.append(flip_pose(pose, axis=0))  # المقلوب

print(f"قبل: {len(original_dataset)} فيديو")
print(f"بعد: {len(augmented_dataset)} فيديو")
# النتيجة: 3 → 6 فيديوهات

# مفيد جداً للإشارات المتماثلة
```

**⚠️ تحذير مهم:**

```python
# لا تقلب إشارات لها معنى اتجاهي!

# ✅ آمن للقلب:
# - "مرحبا" (التلويح)
# - "شكراً"
# - "نعم" (الإيماء)

# ❌ خطر القلب:
# - "يمين" ← يصبح "يسار"!
# - "شرق" ← يصبح "غرب"!
# - أي إشارة تعتمد على الاتجاه

# الحل: فحص يدوي أو قائمة إشارات آمنة
safe_to_flip = ['hello', 'thanks', 'yes', 'no']
if sign_name in safe_to_flip:
    flipped = flip_pose(pose, axis=0)
```

---

## 9. التحويل بين Backends

### 🎯 ما هي Backends ولماذا نحتاج التحويل؟

**Backend** = المكتبة التي تستخدمها لتخزين البيانات والعمليات

**المكتبات الرئيسية:**
1. **NumPy**: الافتراضي، عام الغرض
2. **PyTorch**: للتعلم العميق (Facebook)
3. **TensorFlow**: للتعلم العميق (Google)

---

### 9.1 التحويل إلى PyTorch

```python
def convert_to_torch(pose: Pose):
    torch_pose = pose.torch()
    return torch_pose
```

**مثال - تدريب نموذج PyTorch:**

```python
import torch
import torch.nn as nn

# 1. تحميل البيانات
pose = load_pose('training_data.pose')

# 2. التحويل إلى PyTorch
torch_pose = convert_to_torch(pose)

# 3. استخراج البيانات كـ tensor
X = torch_pose.body.data  # الآن PyTorch tensor!

# 4. استخدامه في نموذج
class SignLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=225, hidden_size=128)
        self.fc = nn.Linear(128, 10)  # 10 إشارات
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = SignLanguageModel()
output = model(X)
```

---

### 9.2 التحويل إلى TensorFlow

```python
def convert_to_tensorflow(pose: Pose):
    tf_pose = pose.tensorflow()
    return tf_pose
```

**مثال - تدريب نموذج TensorFlow:**

```python
import tensorflow as tf

# 1. تحميل البيانات
pose = load_pose('training_data.pose')

# 2. التحويل إلى TensorFlow
tf_pose = convert_to_tensorflow(pose)

# 3. استخراج البيانات كـ tensor
X = tf_pose.body.data  # الآن TensorFlow tensor!

# 4. بناء نموذج Keras
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(None, 225)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(X, y)
```

---

### 9.3 متى تستخدم كل backend؟

| Backend | متى تستخدمه |
|---------|-------------|
| **NumPy** (افتراضي) | معالجة عامة، تحليل، تصور |
| **PyTorch** | البحث العلمي، تطوير نماذج جديدة، مرونة عالية |
| **TensorFlow** | الإنتاج، الخدمات السحابية، التوزيع |

**نصيحة:**
- ابدأ بـ NumPy للتجربة والتحليل
- انتقل إلى PyTorch/TensorFlow عند التدريب

---

## 10. دوال مساعدة إضافية

### 10.1 Focus (ضبط المنظور)

```python
def focus_pose(pose: Pose) -> Pose:
    pose_copy = pose.copy()
    pose_copy.focus()
    return pose_copy
```

**ما الذي يفعله focus():**
1. ينقل جميع النقاط بحيث تبدأ من (0, 0)
2. يوسع النطاق ليملأ المساحة المتاحة

**مثال - تحضير للعرض:**

```python
# الأصل: الشخص في زاوية الصورة، صغير
original = load_pose('corner_video.pose')

# بعد Focus: الشخص في المنتصف، يملأ الشاشة
focused = focus_pose(original)

# مثالي للعرض والتصوير
visualize(focused)
```

---

### 10.2 Slice (قص الإطارات)

```python
def slice_pose(pose: Pose, start: int = 0, end: int = None, step: int = 1) -> Pose:
    if end is None:
        end = pose.body.data.shape[0]
    
    sliced = pose.slice_step(start=start, end=end, step=step)
    return sliced
```

**أمثلة:**

#### مثال 1: استخراج جزء من الفيديو

```python
# فيديو طويل (300 إطار)
full_video = load_pose('long_sign.pose')

# استخراج أول 5 ثواني (بافتراض 25 FPS)
first_5_seconds = slice_pose(full_video, start=0, end=125)

# استخراج من الثانية 5 إلى 10
middle_part = slice_pose(full_video, start=125, end=250)

# آخر 50 إطار
last_part = slice_pose(full_video, start=-50, end=None)
```

#### مثال 2: تقليل FPS يدوياً

```python
# أخذ كل إطار ثاني (تقليل FPS بمقدار النصف)
every_second_frame = slice_pose(pose, step=2)

# أخذ كل إطار ثالث (تقليل إلى الثلث)
every_third_frame = slice_pose(pose, step=3)

print(f"الأصل: {pose.body.data.shape[0]} إطار")
print(f"كل ثاني: {every_second_frame.body.data.shape[0]} إطار")
print(f"كل ثالث: {every_third_frame.body.data.shape[0]} إطار")
```

#### مثال 3: تحليل إطارات محددة

```python
# تحليل كل 10 إطارات فقط (تسريع المعالجة)
sampled = slice_pose(pose, step=10)

# مفيد للتحليل السريع أو المعاينة
```

---

### 10.3 Flatten (تسطيح البيانات)

```python
def flatten_pose_data(pose: Pose) -> np.ndarray:
    flat = pose.body.flatten()
    return flat
```

**ما هو التسطيح؟**

```python
# قبل التسطيح:
# شكل البيانات: [frames, people, points, dimensions]
#               [100, 1, 75, 3]

pose = load_pose('sign.pose')
print(pose.body.data.shape)  # [100, 1, 75, 3]

# بعد التسطيح:
# صف لكل نقطة في كل إطار
flat = flatten_pose_data(pose)
print(flat.shape)  # [N, 7]

# الأعمدة: [frame, person, point, confidence, x, y, z]
```

**مثال - تصدير لـ CSV:**

```python
import pandas as pd

# تسطيح البيانات
flat = flatten_pose_data(pose)

# تحويل إلى DataFrame
df = pd.DataFrame(flat, columns=[
    'frame', 'person', 'point', 'confidence', 'x', 'y', 'z'
])

# حفظ كـ CSV
df.to_csv('pose_data.csv', index=False)

# الآن يمكن فتحه في Excel أو أي برنامج
```

**مثال - تحليل إحصائي:**

```python
# تسطيح البيانات
flat = flatten_pose_data(pose)

# حذف النقاط ذات ثقة منخفضة
high_confidence = flat[flat[:, 3] > 0.5]  # confidence > 0.5

# حساب إحصائيات
mean_x = np.mean(high_confidence[:, 4])
std_y = np.std(high_confidence[:, 5])

print(f"متوسط X: {mean_x}")
print(f"انحراف Y: {std_y}")
```

---

## 11. خط معالجة شامل

### 🎯 دمج كل شيء معاً

```python
def complete_preprocessing_pipeline(pose_path: str, output_path: str):
    """
    خط معالجة كامل جاهز للتدريب
    """
    
    # 1. تحميل
    pose = load_pose(pose_path)
    
    # 2. استخراج المكونات المهمة
    pose = pose.get_components([
        'POSE_LANDMARKS',
        'LEFT_HAND_LANDMARKS',
        'RIGHT_HAND_LANDMARKS'
    ])
    
    # 3. ملء الإطارات المفقودة
    pose = pose.interpolate(new_fps=None, kind='linear')
    
    # 4. التطبيع
    pose.normalize()
    
    # 5. توحيد FPS
    pose = pose.interpolate(new_fps=25, kind='cubic')
    
    # 6. حفظ
    with open(output_path, 'wb') as f:
        pose.write(f)
    
    return pose
```

**سيناريو كامل - من الفيديو إلى النموذج:**

```python
import os

# ═══════════════════════════════════════════════
# المرحلة 1: استخراج Poses من الفيديوهات
# ═══════════════════════════════════════════════

video_files = [
    'videos/sign1.mp4',
    'videos/sign2.mp4',
    'videos/sign3.mp4',
]

# استخدم 01_extract_landmarks_from_video.ipynb
# النتيجة: ملفات .pose

# ═══════════════════════════════════════════════
# المرحلة 2: المعالجة والتنظيف
# ═══════════════════════════════════════════════

pose_files = [
    'data/sign1.pose',
    'data/sign2.pose',
    'data/sign3.pose',
]

processed_poses = []

for pose_file in pose_files:
    # معالجة كل ملف
    processed = complete_preprocessing_pipeline(
        pose_file,
        f'processed/{os.path.basename(pose_file)}'
    )
    processed_poses.append(processed)

print(f"تمت معالجة {len(processed_poses)} ملف")

# ═══════════════════════════════════════════════
# المرحلة 3: تضخيم البيانات
# ═══════════════════════════════════════════════

augmented_dataset = []

for pose in processed_poses:
    # 1. الأصل
    augmented_dataset.append(pose)
    
    # 2. قلب أفقي
    augmented_dataset.append(pose.flip(axis=0))
    
    # 3. تضخيم 2D (5 نسخ)
    for i in range(5):
        aug = pose.copy().augment2d(
            rotation_std=0.2,
            shear_std=0.1,
            scale_std=0.15
        )
        augmented_dataset.append(aug)
    
    # 4. Frame dropout (3 نسخ)
    for i in range(3):
        dropped, _ = pose.frame_dropout_uniform(
            dropout_min=0.6,
            dropout_max=0.9
        )
        augmented_dataset.append(dropped)

print(f"قبل التضخيم: {len(processed_poses)} فيديو")
print(f"بعد التضخيم: {len(augmented_dataset)} فيديو")
# النتيجة: 3 → 3 × (1 + 1 + 5 + 3) = 30 فيديو!

# ═══════════════════════════════════════════════
# المرحلة 4: التحضير للتدريب
# ═══════════════════════════════════════════════

# التحويل إلى PyTorch
X_train = []
for pose in augmented_dataset:
    torch_pose = pose.torch()
    X_train.append(torch_pose.body.data)

# التدريب
# model.fit(X_train, y_train)

print("✅ البيانات جاهزة للتدريب!")
```

---

## 📊 جدول ملخص الميزات

| الميزة | الدالة | الغرض | مثال استخدام |
|--------|--------|-------|---------------|
| **التطبيع الأساسي** | `pose.normalize()` | توحيد الحجم والموقع | مقارنة فيديوهات من كاميرات مختلفة |
| **التطبيع المخصص** | `normalize_pose_custom()` | تطبيع بنقاط مخصصة | تحليل إشارات اليد |
| **التطبيع الإحصائي** | `normalize_distribution()` | Mean=0, Std=1 | إدخال نماذج التعلم الآلي |
| **التضخيم 2D** | `pose.augment2d()` | تدوير، قص، تحجيم | زيادة بيانات التدريب |
| **تغيير FPS** | `pose.interpolate()` | توحيد معدل الإطارات | توحيد السرعة |
| **ملء المفقود** | `interpolate(new_fps=None)` | ملء الإطارات المفقودة | تنظيف البيانات |
| **استخراج مكونات** | `pose.get_components()` | اختيار مكونات محددة | اليدين فقط |
| **إزالة مكونات** | `pose.remove_components()` | حذف مكونات | إزالة الوجه للخصوصية |
| **Bounding Box** | `pose.bbox()` | حساب حدود المكونات | تتبع موقع اليد |
| **Frame Dropout** | `frame_dropout_uniform()` | حذف إطارات عشوائية | تدريب نموذج قوي |
| **القلب** | `pose.flip()` | انعكاس أفقي/رأسي | مضاعفة البيانات |
| **PyTorch** | `pose.torch()` | تحويل لـ PyTorch | تدريب نماذج PyTorch |
| **TensorFlow** | `pose.tensorflow()` | تحويل لـ TensorFlow | تدريب نماذج TensorFlow |
| **Focus** | `pose.focus()` | ضبط المنظور | تحضير للعرض |
| **Slice** | `pose.slice_step()` | قص إطارات | استخراج جزء من الفيديو |
| **Flatten** | `pose.body.flatten()` | تسطيح البيانات | تصدير CSV |

---

## 🎓 أفضل الممارسات (Best Practices)

### للتدريب (Training)

```python
# ✅ افعل
1. طبّع دائماً قبل التدريب
2. وحّد FPS لجميع الفيديوهات
3. استخدم التضخيم لزيادة البيانات
4. احذف المكونات غير الضرورية

# ❌ لا تفعل
1. لا تدرب على بيانات غير منظفة
2. لا تخلط FPS مختلفة
3. لا تضخّم بشكل مبالغ (قد يضر)
4. لا تحتفظ بجميع المكونات دون سبب
```

### للأداء (Performance)

```python
# ✅ لتسريع المعالجة
1. استخرج المكونات المهمة فقط
2. قلل FPS إذا كان عالياً جداً
3. استخدم Bounding Box بدلاً من كل النقاط
4. استخدم dtype مناسب (float32 بدلاً من float64)

# ❌ يبطئ المعالجة
1. الاحتفاظ بـ FACE_LANDMARKS إذا لم تحتاجه
2. FPS عالي جداً (60+) بدون داعٍ
3. معالجة كل النقاط للحسابات البسيطة
4. استخدام float64 دائماً
```

### للجودة (Quality)

```python
# ✅ لنتائج أفضل
1. املأ الإطارات المفقودة بالاستيفاء
2. استخدم cubic interpolation لتغيير FPS
3. احفظ mu, std عند التطبيع الإحصائي
4. تحقق من البيانات بعد كل خطوة

# ❌ يقلل الجودة
1. تجاهل الإطارات المفقودة
2. استخدام linear interpolation دائماً
3. عدم حفظ معاملات التطبيع
4. معالجة دفعة واحدة بدون فحص
```

---

## 🔧 أمثلة سيناريوهات واقعية

### سيناريو 1: تطبيق لغة إشارة للأطفال

```python
def process_for_kids_app(video_path):
    """
    معالجة فيديوهات لغة إشارة للأطفال
    - تركيز على اليدين
    - تحمل جودة منخفضة
    - سرعات مختلفة
    """
    
    # 1. تحميل
    pose = load_pose(video_path)
    
    # 2. الجسم واليدين فقط (الأطفال لا يحتاجون الوجه)
    pose = pose.get_components([
        'POSE_LANDMARKS',
        'LEFT_HAND_LANDMARKS',
        'RIGHT_HAND_LANDMARKS'
    ])
    
    # 3. ملء الإطارات المفقودة (الأطفال يتحركون كثيراً)
    pose = pose.interpolate(new_fps=None, kind='cubic')
    
    # 4. تطبيع بالرسغين (اليدين أهم)
    norm_info = pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_WRIST"),
        p2=("POSE_LANDMARKS", "LEFT_WRIST")
    )
    pose.normalize(norm_info)
    
    # 5. FPS منخفض (للأجهزة الضعيفة)
    pose = pose.interpolate(new_fps=20, kind='cubic')
    
    return pose
```

### سيناريو 2: تطبيق تحليل رياضي محترف

```python
def process_for_sports_analysis(video_path):
    """
    معالجة لتحليل الحركات الرياضية
    - دقة عالية
    - FPS عالي
    - تحليل الجسم كامل
    """
    
    # 1. تحميل
    pose = load_pose(video_path)
    
    # 2. الجسم فقط (مع World Landmarks للدقة 3D)
    pose = pose.get_components([
        'POSE_LANDMARKS',
        'POSE_WORLD_LANDMARKS'
    ])
    
    # 3. FPS عالي (تفاصيل دقيقة)
    pose = pose.interpolate(new_fps=60, kind='cubic')
    
    # 4. تطبيع بالوركين (قاعدة ثابتة)
    norm_info = pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_HIP"),
        p2=("POSE_LANDMARKS", "LEFT_HIP")
    )
    pose.normalize(norm_info)
    
    # 5. لا نستخدم تضخيم (نريد البيانات الحقيقية)
    
    return pose
```

### سيناريو 3: تطبيق تواصل سريع

```python
def process_for_quick_communication(video_path):
    """
    معالجة للتواصل الفوري
    - سرعة أهم من الدقة
    - حجم صغير
    - استجابة فورية
    """
    
    # 1. تحميل
    pose = load_pose(video_path)
    
    # 2. نقاط قليلة فقط
    pose = pose.get_components(
        ['POSE_LANDMARKS'],
        points={
            'POSE_LANDMARKS': [
                'NOSE',
                'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                'LEFT_ELBOW', 'RIGHT_ELBOW',
                'LEFT_WRIST', 'RIGHT_WRIST'
            ]
        }
    )
    
    # 3. FPS منخفض جداً
    pose = pose.interpolate(new_fps=15, kind='linear')
    
    # 4. كل إطار ثالث (تسريع إضافي)
    pose = pose.slice_step(step=3)
    
    # 5. تطبيع بسيط
    pose.normalize()
    
    return pose
```

---

## 🎯 الخلاصة

**هذا الملف يغطي:**
- ✅ جميع أدوات معالجة Pose المتقدمة
- ✅ أمثلة واقعية وعملية
- ✅ سيناريوهات استخدام مختلفة
- ✅ أفضل الممارسات والنصائح

**الخطوات القادمة:**
1. جرّب كل دالة على بياناتك
2. اختر المعالجات المناسبة لتطبيقك
3. اصنع خط معالجة (pipeline) مخصص
4. ابدأ التدريب!

**موارد إضافية:**
- 📁 `01_extract_landmarks_from_video.ipynb` - استخراج Poses
- 📁 `02_convert_pose_formats.ipynb` - تحويل الصيغ
- 📁 `03_read_pose_files.ipynb` - قراءة الملفات
- 📁 `04_visualize_pose.ipynb` - التصوير

---

💡 **نصيحة أخيرة:** ابدأ بسيط ثم أضف التعقيد تدريجياً. لا تستخدم كل الميزات مرة واحدة!

🚀 **حظ سعيد في مشروعك!**
