import pandas as pd
import math
import random
from collections import Counter

#cargamos el dataset
df = pd.read_csv("2016-2024_liga_mx.csv")

#mostramos informacion basica del dataset
print("\nInformacion del dataset:")
print(f"- Partidos totales: {len(df)}")
print(f"- Temporadas: {sorted(df['season'].unique())}")
print(f"- Equipos unicos: {len(set(df['home_team'].unique()) | set(df['away_team'].unique()))}")

#analizamos valores faltantes en columnas importantes
important_cols = ['home_goals', 'away_goals', 'home_goals_half_time', 'away_goals_half_time', 'home_win', 'season']
print("\nValores faltantes en columnas importantes:")
for col in important_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"- {col}: {missing} ({missing/len(df)*100:.1f}%)")

df_clean = df[['home_goals_half_time', 'away_goals_half_time', 'season', 'home_win']].dropna()

print(f"\nDatos despues de limpiar: {len(df_clean)} registros")

home_win_counts = df_clean['home_win'].value_counts()
for value, count in home_win_counts.items():
    print(f"- {value}: {count} ({count/len(df_clean)*100:.1f}%)")

#convertimos la etiqueta home_win en binaria
df_clean["home_win_binary"] = df_clean["home_win"].apply(lambda x: 1 if x == True or x == "True" else 0)

print("\nDistribucion despues de conversion binaria:")
binary_counts = df_clean['home_win_binary'].value_counts()
for value, count in binary_counts.items():
    label = "Victoria Local" if value == 1 else "No Victoria Local"
    print(f"- {label}: {count} ({count/len(df_clean)*100:.1f}%)")

print("\nEstadisticas de goles PRIMER TIEMPO:")
print(f"- Promedio goles primer tiempo local: {df_clean['home_goals_half_time'].mean():.2f}")
print(f"- Promedio goles primer tiempo visitante: {df_clean['away_goals_half_time'].mean():.2f}")
print(f"- Diferencia promedio: {(df_clean['home_goals_half_time'] - df_clean['away_goals_half_time']).mean():.2f}")

#preparamos el dataset final
df_knn = df_clean[['home_goals_half_time', 'away_goals_half_time', 'season', 'home_win_binary']]

#convertimos a lista de registros
dataset = df_knn.values.tolist()

print("\nDIVISION DE DATOS")

#mezclamos y dividimos en train y test
random.seed(42)  #para reproducibilidad
random.shuffle(dataset)
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
train_set = dataset[:split_index]
test_set = dataset[split_index:]

print(f"Conjunto de entrenamiento: {len(train_set)} registros")
print(f"Conjunto de prueba: {len(test_set)} registros")

#verificamos distribucion en train y test
train_wins = sum(1 for row in train_set if row[-1] == 1)
test_wins = sum(1 for row in test_set if row[-1] == 1)
print(f"Victorias locales en entrenamiento: {train_wins}/{len(train_set)} ({train_wins/len(train_set)*100:.1f}%)")
print(f"Victorias locales en prueba: {test_wins}/{len(test_set)} ({test_wins/len(test_set)*100:.1f}%)")

class KNN:
    """KNN implementado desde scratch"""
    
    def __init__(self, k=5):
        self.k = k
        self.train_data = None
    
    def fit(self, train_data):
        """Entrena el modelo (almacena los datos)"""
        self.train_data = train_data
    
    def euclidean_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos (sin contar la etiqueta final)"""
        distance = 0
        for i in range(len(point1) - 1):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)
    
    def get_neighbors(self, test_point):
        """Obtiene los k vecinos mas cercanos."""
        if self.train_data is None:
            raise ValueError("el modelo no ha sido entrenado, llama a fit() primero")
        
        distances = []
        for train_point in self.train_data:
            dist = self.euclidean_distance(test_point, train_point)
            distances.append((train_point, dist))
        distances.sort(key=lambda x: x[1])  #ordenamos por distancia
        neighbors = [distances[i][0] for i in range(min(self.k, len(distances)))]
        return neighbors
    
    def predict_single(self, test_point):
        """Predice la clase (0 o 1) de un punto de prueba"""
        neighbors = self.get_neighbors(test_point)
        output_values = [row[-1] for row in neighbors]
        #se usa counter para manejar empates de forma mas robusta
        vote_counts = Counter(output_values)
        prediction = vote_counts.most_common(1)[0][0]
        return prediction
    
    def predict(self, test_data):
        """Predice multiples puntos"""
        predictions = []
        for test_point in test_data:
            prediction = self.predict_single(test_point)
            predictions.append(prediction)
        return predictions
    
    def evaluate(self, test_data):
        """Evalua el modelo en el conjunto de test"""
        predictions = self.predict(test_data)
        actual = [row[-1] for row in test_data]
        
        correct = sum(1 for pred, real in zip(predictions, actual) if pred == real)
        accuracy = correct / len(test_data)
        
        #calculamos metricas adicionales
        tp = sum(1 for pred, real in zip(predictions, actual) if pred == 1 and real == 1)
        tn = sum(1 for pred, real in zip(predictions, actual) if pred == 0 and real == 0)
        fp = sum(1 for pred, real in zip(predictions, actual) if pred == 1 and real == 0)
        fn = sum(1 for pred, real in zip(predictions, actual) if pred == 0 and real == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'predictions': predictions,
            'actual': actual
        }


print("\nOPTIMIZACION DEL PARAMETRO K")
k_values = [1, 3, 5, 6, 7, 8, 9, 11, 15, 21]
best_k = 5
best_accuracy = 0
results = {}

for k in k_values:
    knn = KNN(k=k)
    knn.fit(train_set)
    metrics = knn.evaluate(test_set)
    accuracy = metrics['accuracy']
    results[k] = metrics
    
    print(f"k={k}: Precision = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\nMejor k encontrado: {best_k} con precision de {best_accuracy:.4f}")

print("\nEVALUACIoN FINAL DEL MODELO")

final_metrics = results[best_k]

print(f"Modelo final con k={best_k}:")
print(f"- Precision (Accuracy): {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
print(f"- Precision (Precision): {final_metrics['precision']:.4f} ({final_metrics['precision']*100:.2f}%)")
print(f"- Recall (Sensibilidad): {final_metrics['recall']:.4f} ({final_metrics['recall']*100:.2f}%)")
print(f"- F1-Score: {final_metrics['f1_score']:.4f}")

#matriz de confusion
cm = final_metrics['confusion_matrix']
print("\nMatriz de Confusion:")
print("                    Predicho")
print("                No Gana  Gana")
print(f"Real  No Gana      {cm['tn']}     {cm['fp']}")
print(f"      Gana         {cm['fn']}     {cm['tp']}")

print("\nCOMPARACIoN CON BASELINE")
majority_class = 1 if train_wins > len(train_set)/2 else 0
baseline_accuracy = max(test_wins, len(test_set) - test_wins) / len(test_set)

print(f"Baseline (siempre predecir clase mayoritaria): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Nuestro modelo KNN: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
improvement = final_metrics['accuracy'] - baseline_accuracy
print(f"Mejora sobre baseline: {improvement:.4f} ({improvement*100:.2f} puntos porcentuales)")

print("\nPREDICCIONES CON EQUIPOS REALES")
final_knn = KNN(k=best_k)
final_knn.fit(train_set)

#función para obtener estadisticas historicas de un equipo
def get_team_stats(team_name, df_original):
    home_games = df_original[df_original['home_team'] == team_name]
    away_games = df_original[df_original['away_team'] == team_name]
    
    if len(home_games) == 0 and len(away_games) == 0:
        return None
    
    home_stats = {
        'games': len(home_games.dropna(subset=['home_goals_half_time'])),
        'avg_goals_ht': home_games['home_goals_half_time'].mean() if len(home_games) > 0 else 0,
        'avg_goals_against_ht': home_games['away_goals_half_time'].mean() if len(home_games) > 0 else 0,
        'wins': len(home_games[home_games['home_win'] == True]) if len(home_games) > 0 else 0
    }
 
    away_stats = {
        'games': len(away_games.dropna(subset=['away_goals_half_time'])),
        'avg_goals_ht': away_games['away_goals_half_time'].mean() if len(away_games) > 0 else 0,
        'avg_goals_against_ht': away_games['home_goals_half_time'].mean() if len(away_games) > 0 else 0,
        'wins': len(away_games[away_games['away_win'] == True]) if len(away_games) > 0 else 0
    }
    
    return {
        'home': home_stats,
        'away': away_stats,
        'total_games': home_stats['games'] + away_stats['games']
    }

#funcion para predecir un partido especifico basandose en estadisticas historicas
def predict_team_match(home_team, away_team, season=2024):
    print(f"\nAnalizando: {home_team} (Local) vs {away_team} (Visitante) - {season}")
    
    #obtener estadisticas de ambos equipos
    home_stats = get_team_stats(home_team, df)
    away_stats = get_team_stats(away_team, df)
    
    if home_stats is None:
        print(f"No se encontraron datos para {home_team}")
        return None
    if away_stats is None:
        print(f"No se encontraron datos para {away_team}")
        return None
    
    #usamos promedios historicos como estimacion para primer tiempo
    home_avg_ht = home_stats['home']['avg_goals_ht'] 
    away_avg_ht = away_stats['away']['avg_goals_ht']
    
    print("Estadisticas historicas:")
    print(f"   {home_team} (como local): {home_stats['home']['games']} partidos, {home_avg_ht:.2f} goles/HT promedio")
    print(f"   {away_team} (como visitante): {away_stats['away']['games']} partidos, {away_avg_ht:.2f} goles/HT promedio")
    
    #creamos escenarios basados en los promedios historicos
    scenarios = [
        {
            'name': 'Escenario Promedio',
            'home_ht': round(home_avg_ht),
            'away_ht': round(away_avg_ht),
            'description': 'Basado en promedios historicos exactos'
        },
        {
            'name': 'Escenario Optimista Local',
            'home_ht': min(3, round(home_avg_ht) + 1),
            'away_ht': max(0, round(away_avg_ht) - 1),
            'description': 'El equipo local tiene un buen primer tiempo'
        },
        {
            'name': 'Escenario Pesimista Local',
            'home_ht': max(0, round(home_avg_ht) - 1),
            'away_ht': min(3, round(away_avg_ht) + 1),
            'description': 'El equipo visitante tiene ventaja al descanso'
        }
    ]
    
    print("\nPredicciones para diferentes escenarios:")
    
    for scenario in scenarios:
        #crear datos para predicción
        match_data = [scenario['home_ht'], scenario['away_ht'], season, None]
        
        #hacer predicción
        prediction = final_knn.predict_single(match_data)
        result = "VICTORIA LOCAL" if prediction == 1 else "NO VICTORIA LOCAL"
        
        print(f"\n   {scenario['name']}:")
        print(f"   HT: {home_team} {scenario['home_ht']} - {scenario['away_ht']} {away_team}")
        print(f"   {scenario['description']}")
        print(f"   Predicción final: {result}")
    
    return scenarios

#obtener lista de equipos disponibles en el dataset
available_teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
print(f"\nEquipos disponibles en el dataset ({len(available_teams)}):")
for i, team in enumerate(available_teams, 1):
    print(f"{i:2d}. {team}")

#ejemplos de partidos con equipos populares de Liga MX
popular_matchups = [
    ("Guadalajara Chivas", "Club America"),  # Clásico Nacional
    ("Club America", "U.N.A.M. - Pumas"),   # Clásico Capitalino
    ("Monterrey", "Tigres UANL"),           # Clásico Regiomontano
    ("Cruz Azul", "Guadalajara Chivas"),    # Otro clásico
    ("Toluca", "Santos Laguna")             # Partido tradicional
]

print("\nEJEMPLOS DE PARTIDOS CLÁSICOS:")

for home, away in popular_matchups:
    # Verificar que ambos equipos existen en el dataset
    if home in available_teams and away in available_teams:
        predict_team_match(home, away, 2024)
        print("-" * 50)
    else:
        missing = []
        if home not in available_teams:
            missing.append(home)
        if away not in available_teams:
            missing.append(away)
        print(f"\nNo se puede analizar {home} vs {away}")
        print(f"   Equipos no encontrados: {', '.join(missing)}")

print("\nPREDICCIÓN PERSONALIZADA:")
print("Puedes usar esta función para predecir cualquier partido:")
print("predict_team_match('Equipo_Local', 'Equipo_Visitante', 2024)")
print("\nEjemplo:")
print("predict_team_match('Monterrey', 'Atlas', 2024)")

