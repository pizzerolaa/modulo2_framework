import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("📊 ANÁLISIS COMPLETO: BIAS, VARIANZA Y AJUSTE DEL MODELO KNN")
print("="*80)

# ======================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ======================

print("\n1️⃣ CARGA Y PREPARACIÓN DE DATOS")
print("-" * 50)

# Cargar dataset
df = pd.read_csv("2016-2024_liga_mx.csv")
df_clean = df[['home_goals_half_time', 'away_goals_half_time', 'season', 'home_win', 'home_team', 'away_team']].dropna()
df_clean["home_win_binary"] = df_clean["home_win"].apply(lambda x: 1 if x == True or x == "True" else 0)

print(f"📈 Dataset original: {len(df)} registros")
print(f"📊 Dataset limpio: {len(df_clean)} registros")
print(f"🎯 Clases balanceadas: {df_clean['home_win_binary'].value_counts().to_dict()}")

# Preparar características
X = df_clean[['home_goals_half_time', 'away_goals_half_time', 'season']].values
y = df_clean['home_win_binary'].values

# ======================
# 2. SEPARACIÓN TRAIN/VALIDATION/TEST
# ======================

print("\n2️⃣ SEPARACIÓN DE DATOS: TRAIN/VALIDATION/TEST")
print("-" * 50)

# Primera división: Train+Validation (80%) vs Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Segunda división: Train (60%) vs Validation (20%)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 de 80% = 20% total
)

print(f"🔹 Conjunto de Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"🔹 Conjunto de Validación: {len(X_validation)} muestras ({len(X_validation)/len(X)*100:.1f}%)")
print(f"🔹 Conjunto de Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

# Verificar distribución de clases
print(f"\n📊 DISTRIBUCIÓN DE CLASES:")
print(f"Train - Victoria Local: {np.sum(y_train)}/{len(y_train)} ({np.sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Validation - Victoria Local: {np.sum(y_validation)}/{len(y_validation)} ({np.sum(y_validation)/len(y_validation)*100:.1f}%)")
print(f"Test - Victoria Local: {np.sum(y_test)}/{len(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%)")

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# ======================
# 3. ANÁLISIS DE CURVAS DE VALIDACIÓN
# ======================

print("\n3️⃣ ANÁLISIS DE CURVAS DE VALIDACIÓN")
print("-" * 50)

# Rango de valores k para analizar
k_range = np.arange(1, 51, 2)  # k = 1, 3, 5, ..., 49

# Calcular curvas de validación
train_scores, validation_scores = validation_curve(
    KNeighborsClassifier(), X_train_scaled, y_train,
    param_name='n_neighbors', param_range=k_range,
    cv=5, scoring='accuracy', n_jobs=-1
)

# Calcular medias y desviaciones estándar
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)

# Encontrar mejor k
best_k_idx = np.argmax(validation_mean)
best_k = k_range[best_k_idx]
best_validation_score = validation_mean[best_k_idx]

print(f"🎯 Mejor k encontrado: {best_k}")
print(f"📈 Mejor score de validación: {best_validation_score:.4f} ± {validation_std[best_k_idx]:.4f}")

# ======================
# 4. ANÁLISIS DE CURVAS DE APRENDIZAJE
# ======================

print("\n4️⃣ ANÁLISIS DE CURVAS DE APRENDIZAJE")
print("-" * 50)

# Calcular curvas de aprendizaje con el mejor k
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores_lc, validation_scores_lc = learning_curve(
    KNeighborsClassifier(n_neighbors=best_k), X_train_scaled, y_train,
    train_sizes=train_sizes, cv=5, scoring='accuracy', n_jobs=-1
)

# Calcular medias y desviaciones
train_mean_lc = np.mean(train_scores_lc, axis=1)
train_std_lc = np.std(train_scores_lc, axis=1)
validation_mean_lc = np.mean(validation_scores_lc, axis=1)
validation_std_lc = np.std(validation_scores_lc, axis=1)

# ======================
# 5. EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA
# ======================

print("\n5️⃣ EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA")
print("-" * 50)

# Entrenar modelo final con mejores parámetros
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)

# Evaluaciones en los tres conjuntos
train_accuracy = final_model.score(X_train_scaled, y_train)
validation_accuracy = final_model.score(X_validation_scaled, y_validation)
test_accuracy = final_model.score(X_test_scaled, y_test)

print(f"📊 RESULTADOS FINALES:")
print(f"🔹 Accuracy en Entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"🔹 Accuracy en Validación: {validation_accuracy:.4f} ({validation_accuracy*100:.2f}%)")
print(f"🔹 Accuracy en Prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Predicciones para análisis detallado
y_pred_test = final_model.predict(X_test_scaled)

# ======================
# 6. DIAGNÓSTICO DE BIAS, VARIANZA Y AJUSTE
# ======================

print("\n6️⃣ DIAGNÓSTICO DE BIAS, VARIANZA Y AJUSTE")
print("-" * 50)

# Calcular diferencias para diagnóstico
train_val_gap = train_accuracy - validation_accuracy
val_test_gap = validation_accuracy - test_accuracy

print(f"📊 MÉTRICAS DE DIAGNÓSTICO:")
print(f"🔹 Gap Entrenamiento-Validación: {train_val_gap:.4f} ({train_val_gap*100:.2f}%)")
print(f"🔹 Gap Validación-Prueba: {val_test_gap:.4f} ({val_test_gap*100:.2f}%)")

# Diagnóstico de BIAS
if validation_accuracy < 0.65:
    bias_level = "ALTO"
    bias_explanation = "El modelo tiene dificultades para capturar patrones, incluso en validación"
elif validation_accuracy < 0.75:
    bias_level = "MEDIO"
    bias_explanation = "El modelo captura patrones básicos pero podría mejorar"
else:
    bias_level = "BAJO"
    bias_explanation = "El modelo captura bien los patrones subyacentes de los datos"

# Diagnóstico de VARIANZA
if train_val_gap > 0.15:
    variance_level = "ALTA"
    variance_explanation = "Gran diferencia entre entrenamiento y validación indica alta varianza"
elif train_val_gap > 0.05:
    variance_level = "MEDIA"
    variance_explanation = "Diferencia moderada entre entrenamiento y validación"
else:
    variance_level = "BAJA"
    variance_explanation = "Poca diferencia entre entrenamiento y validación indica baja varianza"

# Diagnóstico de AJUSTE
if train_val_gap > 0.15 and validation_accuracy > 0.70:
    fit_level = "OVERFITTING"
    fit_explanation = "Modelo memoriza entrenamiento pero generaliza mal"
elif validation_accuracy < 0.65:
    fit_level = "UNDERFITTING"
    fit_explanation = "Modelo demasiado simple para capturar patrones complejos"
else:
    fit_level = "GOOD FIT"
    fit_explanation = "Modelo balanceado entre complejidad y generalización"

print(f"\n🔍 DIAGNÓSTICOS:")
print(f"📈 BIAS (Sesgo): {bias_level}")
print(f"   └── {bias_explanation}")
print(f"📉 VARIANZA: {variance_level}")
print(f"   └── {variance_explanation}")
print(f"⚖️ AJUSTE: {fit_level}")
print(f"   └── {fit_explanation}")

# ======================
# 7. ANÁLISIS DE VARIANZA POR CROSS-VALIDATION
# ======================

print("\n7️⃣ ANÁLISIS DE VARIANZA POR CROSS-VALIDATION")
print("-" * 50)

# Cross-validation con diferentes k
k_values_cv = [1, 3, 5, 7, 11, 15, 21, 31]
cv_results = {}

for k in k_values_cv:
    cv_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), 
        X_train_scaled, y_train, cv=10, scoring='accuracy'
    )
    cv_results[k] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"k={k:2d}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (variabilidad: {cv_scores.std()/cv_scores.mean()*100:.1f}%)")

# ======================
# 8. GENERACIÓN DE GRÁFICAS
# ======================

print("\n8️⃣ GENERANDO GRÁFICAS COMPARATIVAS")
print("-" * 50)

# Crear figura con subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Análisis Completo del Modelo KNN: Bias, Varianza y Ajuste', fontsize=16, fontweight='bold')

# Gráfica 1: Curva de Validación
ax1 = axes[0, 0]
ax1.plot(k_range, train_mean, 'o-', color='blue', label='Entrenamiento', alpha=0.8)
ax1.fill_between(k_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
ax1.plot(k_range, validation_mean, 'o-', color='red', label='Validación', alpha=0.8)
ax1.fill_between(k_range, validation_mean - validation_std, validation_mean + validation_std, alpha=0.2, color='red')
ax1.axvline(x=best_k, color='green', linestyle='--', label=f'Mejor k={best_k}')
ax1.set_xlabel('Número de Vecinos (k)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Curva de Validación\n(Análisis de Complejidad)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica 2: Curva de Aprendizaje
ax2 = axes[0, 1]
ax2.plot(train_sizes_abs, train_mean_lc, 'o-', color='blue', label='Entrenamiento', alpha=0.8)
ax2.fill_between(train_sizes_abs, train_mean_lc - train_std_lc, train_mean_lc + train_std_lc, alpha=0.2, color='blue')
ax2.plot(train_sizes_abs, validation_mean_lc, 'o-', color='red', label='Validación', alpha=0.8)
ax2.fill_between(train_sizes_abs, validation_mean_lc - validation_std_lc, validation_mean_lc + validation_std_lc, alpha=0.2, color='red')
ax2.set_xlabel('Tamaño del Conjunto de Entrenamiento')
ax2.set_ylabel('Accuracy')
ax2.set_title('Curva de Aprendizaje\n(Análisis de Datos)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfica 3: Comparación Train/Val/Test
ax3 = axes[0, 2]
accuracies = [train_accuracy, validation_accuracy, test_accuracy]
labels = ['Entrenamiento', 'Validación', 'Prueba']
colors = ['blue', 'orange', 'green']
bars = ax3.bar(labels, accuracies, color=colors, alpha=0.7)
ax3.set_ylabel('Accuracy')
ax3.set_title('Comparación de Accuracy\n(Train/Val/Test)')
ax3.set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Gráfica 4: Matriz de Confusión
ax4 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['No Victoria', 'Victoria'], 
            yticklabels=['No Victoria', 'Victoria'])
ax4.set_title('Matriz de Confusión\n(Conjunto de Prueba)')
ax4.set_xlabel('Predicción')
ax4.set_ylabel('Realidad')

# Gráfica 5: Variabilidad por Cross-Validation
ax5 = axes[1, 1]
k_vals = list(cv_results.keys())
means = [cv_results[k]['mean'] for k in k_vals]
stds = [cv_results[k]['std'] for k in k_vals]
ax5.errorbar(k_vals, means, yerr=stds, fmt='o-', capsize=5, capthick=2, color='purple')
ax5.set_xlabel('Número de Vecinos (k)')
ax5.set_ylabel('Accuracy (Cross-Validation)')
ax5.set_title('Variabilidad del Modelo\n(10-Fold Cross-Validation)')
ax5.grid(True, alpha=0.3)

# Gráfica 6: Diagnóstico Visual
ax6 = axes[1, 2]
# Crear gráfica de radar para diagnóstico
categories = ['Bias', 'Varianza', 'Ajuste']
values = []

# Convertir diagnósticos a valores numéricos (0-1, donde 1 es mejor)
bias_score = 1 - (0.8 if bias_level == "ALTO" else 0.5 if bias_level == "MEDIO" else 0.2)
variance_score = 1 - (0.8 if variance_level == "ALTA" else 0.5 if variance_level == "MEDIA" else 0.2)
fit_score = 0.8 if fit_level == "GOOD FIT" else 0.3

values = [bias_score, variance_score, fit_score]
colors_diag = ['red' if v < 0.4 else 'orange' if v < 0.7 else 'green' for v in values]

bars_diag = ax6.bar(categories, values, color=colors_diag, alpha=0.7)
ax6.set_ylabel('Score (1 = Mejor)')
ax6.set_title('Diagnóstico del Modelo\n(Bias/Varianza/Ajuste)')
ax6.set_ylim(0, 1)
for bar, val, cat in zip(bars_diag, values, categories):
    level = bias_level if cat == 'Bias' else variance_level if cat == 'Varianza' else fit_level
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             level, ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('analisis_knn_completo.png', dpi=300, bbox_inches='tight')
print("📈 Gráficas guardadas en 'analisis_knn_completo.png'")

# ======================
# 9. REPORTE FINAL
# ======================

print("\n" + "="*80)
print("📋 REPORTE FINAL DE ANÁLISIS")
print("="*80)

print(f"""
🎯 RESULTADOS PRINCIPALES:
• Mejor k encontrado: {best_k} vecinos
• Accuracy en Entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)
• Accuracy en Validación: {validation_accuracy:.4f} ({validation_accuracy*100:.2f}%)
• Accuracy en Prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)

🔍 DIAGNÓSTICOS:
• BIAS (Sesgo): {bias_level}
  └── {bias_explanation}
  
• VARIANZA: {variance_level}
  └── {variance_explanation}
  
• AJUSTE: {fit_level}
  └── {fit_explanation}

📊 MÉTRICAS DE CALIDAD:
• Gap Entrenamiento-Validación: {train_val_gap:.4f} ({train_val_gap*100:.2f}%)
• Gap Validación-Prueba: {val_test_gap:.4f} ({val_test_gap*100:.2f}%)
• Variabilidad Cross-Validation: {cv_results[best_k]['std']:.4f}

🎖️ CONCLUSIÓN GENERAL:
El modelo KNN con k={best_k} muestra un {fit_level.lower()}, con {bias_level.lower()} bias y 
{variance_level.lower()} varianza. {'Es adecuado para producción.' if fit_level == 'GOOD FIT' else 'Requiere ajustes adicionales.'}
""")

# Guardar resultados en archivo
with open('analisis_resultados.txt', 'w', encoding='utf-8') as f:
    f.write(f"""ANÁLISIS COMPLETO DEL MODELO KNN
================================

CONFIGURACIÓN:
- Dataset: 2016-2024_liga_mx.csv
- Registros: {len(df_clean)}
- Características: 3 (goles primer tiempo + temporada)
- División: 60% Train / 20% Validation / 20% Test

RESULTADOS:
- Mejor k: {best_k}
- Accuracy Entrenamiento: {train_accuracy:.4f}
- Accuracy Validación: {validation_accuracy:.4f}
- Accuracy Prueba: {test_accuracy:.4f}

DIAGNÓSTICOS:
- Bias: {bias_level} - {bias_explanation}
- Varianza: {variance_level} - {variance_explanation}
- Ajuste: {fit_level} - {fit_explanation}

MÉTRICAS:
- Gap Train-Val: {train_val_gap:.4f}
- Gap Val-Test: {val_test_gap:.4f}
- Std CV: {cv_results[best_k]['std']:.4f}
""")

print("\n✅ Análisis completado. Archivos generados:")
print("  📈 analisis_knn_completo.png (gráficas)")
print("  📄 analisis_resultados.txt (reporte)")
print("\n" + "="*80)