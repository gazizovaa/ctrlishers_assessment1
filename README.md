# **CTRL IS HERS** Competition

Bu repo binary classification (ikili təsnifat) problemini əhatə edir. Layihənin məqsədi verilmiş məlumatlar əsasında `target` dəyişəninin ehtimalını proqnozlaşdırmaqdır. Nəticələr **ROC-AUC** metrikası ilə qiymətləndirilir.

---

## İstifadə olunan texnologiyalar

- **Python 3**
- `pandas`, `numpy` – məlumatların emalı və təhlili
- `scikit-learn` – preprocessing, pipeline və modelin qurulması
- (gələcək təcrübələr üçün: `xgboost`, `lightgbm`, `catboost`, `shap` istifadə oluna bilər)

---

## Kitabxanaların quraşdırılması

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost shap
```

---

## Layihənin işə salınması

1. Repo-nu klonlayın:
   ```bash
   git clone <repo-linki>
   cd <repo-adı>
   ```
2. Notebook-u Kaggle, Colab və ya Jupyter-də açın.
3. Hüceyrələri (cells) yuxarıdan aşağıya ardıcıl icra edin.
4. Sonda `sample2.csv` faylı yaranacaq – bunu Kaggle-a submit edə bilərsiniz.

---

## Addım-addım izahlar

### Məlumatların yüklənməsi

Kaggle API vasitəsilə müsabiqə məlumatları yüklənir və arxiv açılır:

```bash
kaggle competitions download -c ctrl-is-hers-final
unzip ctrl-is-hers-final.zip
```

Sonra `train.csv` və `test.csv` pandas ilə oxunur. Test faylındakı `id` sütunu sonda submission üçün ayrıca saxlanılır.

### EDA

- `df.shape` və `df.info()` ilə sətir/sütun sayı və data tipləri yoxlanılır.
- Hər sütun üçün **boşluq faizi**, **unikal dəyər sayı** və **nümunə dəyərlər** göstərən xülasə cədvəli hazırlanır.

### Feature Engineering-in tətbiq edilməsi

`publish_date` sütunu modelin başa düşə bilməsi üçün ayrı-ayrı rəqəmlərə bölünür:

| Yeni sütun | Mənası |
|---|---|
| `publish_year` | İl |
| `publish_month` | Ay |
| `publish_day` | Gün |

Bu əməliyyat `add_date_features()` funksiyası ilə həm train, həm də test datasına tətbiq olunur. Sonda orijinal `publish_date` sütunu silinir.

### Lazımsız sütunların silinməsi

- `id` – yalnız identifikatordur, modelə fayda vermir.
- `weekday` – silinir, çünki feature engineering tətbiq etdikdən sonra əldə etdiyim sütunlardan `publish_month` ilə oxşarlıq təşkil edir. Bu sütun özündə kateqorik dəyərlər daşıyır, amma modelimiz rəqəmlərlə işləmək istədiyi üçün avtomatik olaraq bunu silib digərini seçirik.

### Train / Validation bölgüsü

Data **80% təlim / 20% yoxlama** nisbətində bölünür:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

`stratify=y` – hər iki hissədə `target` siniflərinin nisbətinin eyni qalmasını təmin edir.

### Preprocessing (Pipeline-ın qurulması)

Sütunları tipinə görə iki qrupa bölür və fərqli emal edirik:

**Rəqəmsal sütunlar:**
1. Boş dəyərlər **median** ilə doldurulur, çünki median kənar dəyərlərə qarşı həssas deyil. İstifadə etdiyim model kənar dəyərlərə həssas olduğu üçün median-ı göstərici kimi istifadə edirik.
2. `StandardScaler` ilə miqyaslanır, çünki istifadə etdiyim model miqyasa həssasdır.

**Kateqoriyalı (mətn) sütunlar:**
1. Boş dəyərlər **ən çox təkrarlanan dəyər** ilə doldurulur.
2. `OneHotEncoder` ilə rəqəmlərə çevrilir.

Hər iki qrup `ColumnTransformer` ilə birləşdirilir. Önəmli qayda odur ki, preprocessing yalnız **train** üzərində `fit` olunur, validation və test üzərində isə yalnız `transform` edilir. Bu, data leakage (məlumat sızması) probleminin qarşısını alır.

`remainder='passthrough'` ilə rəqəmsal və kateqoriyalı sütunlarla əlaqəsi olmayan sütunları dəyişdirmədən modelə ötürə bilərik.

### Baseline model – Logistic Regression

İlk (baza) model olaraq `LogisticRegression(max_iter=1000)` seçilib və train datasında öyrədilib. Modelin dəqiqliyi həm train, həm də validation üzərində `score()` ilə yoxlanılır.

Bu modeli seçməyimin səbəbləri onun sürətli və izah oluna bilən olmasıdır. Digər tərəfdən, ROC-AUC metrikası üçün uyğundur, çünki birbaşa sinif ehtimallarını verir. Həmçinin digər mürəkkəb modellərlə müqayisədə bu model bir istinad nöqtəsi yaradır.

### Qiymətləndirmə

Validation datası üçün sinif ehtimalları hesablanır və **ROC-AUC** ölçülür:

```python
y_proba = base_model.predict_proba(X_val_preprocessed)[:, 1]
roc_auc_score(y_val, y_proba)
```

### Test üçün proqnoz və submission

- Test datası eyni preprocessing-dən keçirilir.
- Model hər sətir üçün `target` ehtimalını proqnozlaşdırır.
- `id` və `target` sütunlarından ibarət fayl yaradılır və `sample2.csv` kimi saxlanılır:

```python
submission = pd.DataFrame({'id': test_ids, 'target': test_proba})
submission.to_csv('sample2.csv', index=False)
```
