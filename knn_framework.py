import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("=== IMPLEMENTACIÓN KNN CON SCIKIT-LEARN ===\n")

#cargar dataset
df = pd.read_csv("2016-2024_liga_mx.csv")

print("INFORMACIÓN DEL DATASET:")
print(f"- Partidos totales: {len(df)}")
print(f"- Temporadas: {sorted(df['season'].unique())}")
print(f"- Equipos únicos: {len(set(df['home_team'].unique()) | set(df['away_team'].unique()))}")

#limpieza de datos
important_cols = ['home_goals', 'away_goals', 'home_goals_half_time', 'away_goals_half_time', 'home_win', 'season']
print("\nVALORES FALTANTES:")
for col in important_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"- {col}: {missing} ({missing/len(df)*100:.1f}%)")

df_clean = df[['home_goals_half_time', 'away_goals_half_time', 'season', 'home_win']].dropna()
print(f"\nDatos después de limpiar: {len(df_clean)} registros")

df_clean["home_win_binary"] = df_clean["home_win"].apply(lambda x: 1 if x == True or x == "True" else 0)

print("\nDISTRIBUCIÓN DE CLASES:")
binary_counts = df_clean['home_win_binary'].value_counts()
for value, count in binary_counts.items():
    label = "Victoria Local" if value == 1 else "No Victoria Local"
    print(f"- {label}: {count} ({count/len(df_clean)*100:.1f}%)")

print("\nESTADÍSTICAS PRIMER TIEMPO:")
print(f"- Promedio goles primer tiempo local: {df_clean['home_goals_half_time'].mean():.2f}")
print(f"- Promedio goles primer tiempo visitante: {df_clean['away_goals_half_time'].mean():.2f}")
print(f"- Diferencia promedio: {(df_clean['home_goals_half_time'] - df_clean['away_goals_half_time']).mean():.2f}")

#preparar características (X) y variable objetivo (y)
X = df_clean[['home_goals_half_time', 'away_goals_half_time', 'season']].values
y = df_clean['home_win_binary'].values

print("\nCARACTERÍSTICAS PREPARADAS:")
print(f"- Forma de X: {X.shape}")
print(f"- Forma de y: {y.shape}")

#división train/test
y_np = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_np
)

print("\nDIVISIÓN DE DATOS:")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print(f"Victorias locales en entrenamiento: {np.sum(y_train)}/{len(y_train)} ({np.sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Victorias locales en prueba: {np.sum(y_test)}/{len(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%)")

print("\nNORMALIZACIÓN DE CARACTERÍSTICAS:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Características normalizadas usando StandardScaler")
print(f"- Media entrenamiento después de escalar: {np.mean(X_train_scaled, axis=0)}")
print(f"- Desviación estándar entrenamiento: {np.std(X_train_scaled, axis=0)}")

print("\n🔧 OPTIMIZACIÓN DE HIPERPARÁMETROS:")

#definimos un rango de valores k para probar
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 25, 31],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

#grid search con validación cruzada
knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
print(f"- k (n_neighbors): {grid_search.best_params_['n_neighbors']}")
print(f"- Pesos: {grid_search.best_params_['weights']}")
print(f"- Métrica: {grid_search.best_params_['metric']}")
print(f"- Mejor score CV: {grid_search.best_score_:.4f}")

#modelo final con mejores parámetros
best_knn = grid_search.best_estimator_
print("\nMODELO FINAL ENTRENADO:")
print(f"Modelo: {best_knn}")

#predicciones
y_pred = best_knn.predict(X_test_scaled)

#metricas
accuracy = accuracy_score(y_test, y_pred)
print("\nEVALUACIÓN DEL MODELO:")
print(f"Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

#reporte de clasificacion detallado
print("\nREPORTE DE CLASIFICACIÓN:")
target_names = ['No Victoria Local', 'Victoria Local']
print(classification_report(y_test, y_pred, target_names=target_names))

#matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("MATRIZ DE CONFUSIÓN:")
print("                    Predicho")
print("                No Gana  Gana")
print(f"Real  No Gana      {cm[0,0]}     {cm[0,1]}")
print(f"      Gana         {cm[1,0]}     {cm[1,1]}")

print("\nVALIDACIÓN CRUZADA (5-FOLD):")
cv_scores = cross_val_score(best_knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Scores individuales: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Precisión promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#predictor simple con baseline
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\nCOMPARACIÓN CON BASELINE:")
print(f"Baseline (clase mayoritaria): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Nuestro modelo KNN: {accuracy:.4f} ({accuracy*100:.2f}%)")
improvement = accuracy - baseline_accuracy
print(f"Mejora sobre baseline: {improvement:.4f} ({improvement*100:.2f} puntos porcentuales)")

print("\nCOMPARACIÓN: SCRATCH vs FRAMEWORK:")
print("Tu implementación desde scratch: ~77.78% (k=7)")
print(f"Scikit-learn optimizado: {accuracy*100:.2f}% (k={grid_search.best_params_['n_neighbors']})")

def get_team_stats_framework(team_name):
    """Obtiene estadísticas de un equipo específico"""
    team_home = df_clean[df_clean.index.isin(df[df['home_team'] == team_name].index)]
    team_away = df_clean[df_clean.index.isin(df[df['away_team'] == team_name].index)]
    
    if len(team_home) == 0 and len(team_away) == 0:
        return None
    
    #estadísticas como local
    home_stats = {
        'games': len(team_home),
        'avg_goals_ht': team_home['home_goals_half_time'].mean() if len(team_home) > 0 else 0,
    }
    
    #estadísticas como visitante  
    away_stats = {
        'games': len(team_away),
        'avg_goals_ht': team_away['away_goals_half_time'].mean() if len(team_away) > 0 else 0,
    }
    
    return {
        'home': home_stats,
        'away': away_stats,
        'total_games': home_stats['games'] + away_stats['games']
    }

def predict_team_match_framework(home_team, away_team, season=2024):
    """Predice un partido específico usando el modelo entrenado"""
    print(f"\nPREDICCIÓN: {home_team} vs {away_team} ({season})")
    
    # Obtener estadísticas
    home_stats = get_team_stats_framework(home_team)
    away_stats = get_team_stats_framework(away_team)
    
    if home_stats is None or away_stats is None:
        print("Uno o ambos equipos no encontrados en el dataset")
        return
    
    print("\nESTADÍSTICAS HISTÓRICAS:")
    print(f"{home_team} (Local):")
    print(f"  - Partidos como local: {home_stats['home']['games']}")
    print(f"  - Promedio goles primer tiempo (local): {home_stats['home']['avg_goals_ht']:.2f}")
    
    print(f"{away_team} (Visitante):")
    print(f"  - Partidos como visitante: {away_stats['away']['games']}")
    print(f"  - Promedio goles primer tiempo (visitante): {away_stats['away']['avg_goals_ht']:.2f}")
    
    #crear escenarios de predicción
    scenarios = [
        {
            'name': 'Promedio Histórico',
            'home_ht': home_stats['home']['avg_goals_ht'],
            'away_ht': away_stats['away']['avg_goals_ht']
        },
        {
            'name': 'Escenario Optimista Local',
            'home_ht': min(3, home_stats['home']['avg_goals_ht'] + 0.5),
            'away_ht': max(0, away_stats['away']['avg_goals_ht'] - 0.3)
        },
        {
            'name': 'Escenario Pesimista Local',
            'home_ht': max(0, home_stats['home']['avg_goals_ht'] - 0.3),
            'away_ht': min(3, away_stats['away']['avg_goals_ht'] + 0.5)
        }
    ]
    
    print("\nPREDICCIONES:")
    for scenario in scenarios:
        #crear características para predicción
        features = np.array([[scenario['home_ht'], scenario['away_ht'], season]])
        features_scaled = scaler.transform(features)
        
        #hacer predicción
        prediction = best_knn.predict(features_scaled)[0]
        probability = best_knn.predict_proba(features_scaled)[0]
        
        result = "Victoria Local" if prediction == 1 else "No Victoria Local"
        confidence = probability[prediction] * 100
        
        print(f"  {scenario['name']}: {scenario['home_ht']:.1f}-{scenario['away_ht']:.1f} → {result} ({confidence:.1f}%)")

#ejemplos de predicciones
print("\nPREDICCIONES CON EQUIPOS REALES:")

equipos_disponibles = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
print(f"Equipos disponibles: {sorted(equipos_disponibles)}")

clasicos = [
    ('Guadalajara Chivas', 'Club America'),
    ('Club America', 'U.N.A.M. - Pumas'),
    ('Monterrey', 'Tigres UANL'),
    ('Cruz Azul', 'Guadalajara Chivas'),
    ('Toluca', 'Atlas')
]

for home, away in clasicos:
    if home in equipos_disponibles and away in equipos_disponibles:
        predict_team_match_framework(home, away, 2024)