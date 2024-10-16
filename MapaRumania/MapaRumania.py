import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import threading

#INTEGRANTES:
#-Vidaurre Serpa Abraham
#-Zavaleta Marcelo Roger

# Representación del mapa de Rumania
romania_map = {
    'Arad': [('Zerind', 75), ('Timisoara', 118), ('Sibiu', 140)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
    'Giurgiu': [('Bucharest', 90)],
    'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Eforie': [('Hirsova', 86)],
    'Vaslui': [('Urziceni', 142), ('Iasi', 92)],
    'Iasi': [('Vaslui', 92), ('Neamt', 87)],
    'Neamt': [('Iasi', 87)],
}

# Función para cargar y mostrar la imagen del mapa
def load_map_image():
    map_image = Image.open("recursos/rumania.png")
    map_image = map_image.resize((1000, 600), Image.Resampling.LANCZOS)
    map_photo = ImageTk.PhotoImage(map_image)
    map_label.config(image=map_photo)
    map_label.image = map_photo

# Función para actualizar la tabla de frontera
def update_frontera_table(frontera):
    for row in frontera_table.get_children():
        frontera_table.delete(row)
    for i, city in enumerate(frontera):
        frontera_table.insert('', 'end', values=(i + 1, city))
    window.update_idletasks()

# Función para actualizar la tabla de explorados
def update_explorados_table(explorados):
    for row in explorados_table.get_children():
        explorados_table.delete(row)
    for i, city in enumerate(explorados):
        explorados_table.insert('', 'end', values=(i + 1, city))
    window.update_idletasks()

# Función para calcular heurística dinámica con respecto a la ciudad de destino
def calculate_heuristic(graph, target_city):
    new_heuristics = {}
    for city in graph:
        if city == target_city:
            new_heuristics[city] = 0
        else:
            # Utilizamos la heurística original para simplificar
            new_heuristics[city] = heuristics.get(city, float('inf'))
    return new_heuristics

# Búsqueda en amplitud
def bfs(graph, start):
    visited = set()
    queue = [(start, 0, [start])]
    distances = {start: 0}
    paths = {start: [start]}
    frontera = [start]
    explorados = []
    
    while queue:
        update_frontera_table(frontera)
        update_explorados_table(explorados)

        current_city, current_distance, path = queue.pop(0)
        if current_city not in visited:
            visited.add(current_city)
            explorados.append(current_city)
            if current_city in frontera:
                frontera.remove(current_city)

            for neighbor, distance in graph[current_city]:
                if neighbor not in visited:
                    cumulative_distance = current_distance + distance
                    if neighbor not in distances or cumulative_distance < distances[neighbor]:
                        distances[neighbor] = cumulative_distance
                        paths[neighbor] = path + [neighbor]
                    queue.append((neighbor, cumulative_distance, path + [neighbor]))
                    if neighbor not in frontera:
                        frontera.append(neighbor)
    
    return distances, paths

# Búsqueda en profundidad
def dfs(graph, start):
    visited = set()
    stack = [(start, 0, [start])]
    distances = {start: 0}
    paths = {start: [start]}
    frontera = [start]
    explorados = []
    
    while stack:
        update_frontera_table(frontera)
        update_explorados_table(explorados)

        current_city, current_distance, path = stack.pop()
        if current_city not in visited:
            visited.add(current_city)
            explorados.append(current_city)
            if current_city in frontera:
                frontera.remove(current_city)

            for neighbor, distance in graph[current_city]:
                if neighbor not in visited:
                    cumulative_distance = current_distance + distance
                    if neighbor not in distances or cumulative_distance < distances[neighbor]:
                        distances[neighbor] = cumulative_distance
                        paths[neighbor] = path + [neighbor]
                    stack.append((neighbor, cumulative_distance, path + [neighbor]))
                    if neighbor not in frontera:
                        frontera.append(neighbor)
    
    return distances, paths

# Búsqueda en profundidad limitada
def limited_depth_search(graph, start, limit):
    visited = set()
    stack = [(start, 0, 0, [start])]  # (nodo, distancia acumulada, profundidad, camino)
    distances = {start: 0}
    paths = {start: [start]}
    frontera = [start]
    explorados = []
    
    while stack:
        update_frontera_table(frontera)
        update_explorados_table(explorados)

        current_city, current_distance, depth, path = stack.pop()
        if current_city not in visited and depth <= limit:
            visited.add(current_city)
            explorados.append(current_city)
            if current_city in frontera:
                frontera.remove(current_city)

            for neighbor, distance in graph[current_city]:
                if neighbor not in visited:
                    cumulative_distance = current_distance + distance
                    if neighbor not in distances or cumulative_distance < distances[neighbor]:
                        distances[neighbor] = cumulative_distance
                        paths[neighbor] = path + [neighbor]
                    stack.append((neighbor, cumulative_distance, depth + 1, path + [neighbor]))
                    if neighbor not in frontera:
                        frontera.append(neighbor)
    
    return distances, paths

# Búsqueda voraz con heurística dinámica
def greedy_search(graph, start, target, heuristic):
    # Recalcular heurísticas en función de la ciudad objetivo (target)
    heuristic = calculate_heuristic(graph, target)

    visited = set()
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic.get(start, float('inf')), start, 0, [start]))
    distances = {start: 0}
    paths = {start: [start]}
    frontera = [start]
    explorados = []

    while priority_queue:
        update_frontera_table(frontera)
        update_explorados_table(explorados)

        _, current_city, current_distance, path = heapq.heappop(priority_queue)
        if current_city not in visited:
            visited.add(current_city)
            explorados.append(current_city)
            if current_city in frontera:
                frontera.remove(current_city)

            for neighbor, distance in graph[current_city]:
                if neighbor not in visited:
                    cumulative_distance = current_distance + distance
                    if neighbor not in distances or cumulative_distance < distances[neighbor]:
                        distances[neighbor] = cumulative_distance
                        paths[neighbor] = path + [neighbor]
                    heapq.heappush(priority_queue, (heuristic.get(neighbor, float('inf')), neighbor, cumulative_distance, path + [neighbor]))
                    if neighbor not in frontera:
                        frontera.append(neighbor)
    
    return distances, paths

# Función para dibujar el camino en un grafo con networkx
def draw_path_networkx(path):
    G = nx.Graph()

    # Añadir nodos y bordes (ciudades y distancias)
    for city, neighbors in romania_map.items():
        for neighbor, distance in neighbors:
            G.add_edge(city, neighbor, weight=distance)

    # Posiciones aproximadas de las ciudades
    city_positions = {
        'Arad': (100, 150),
        'Zerind': (80, 100),
        'Oradea': (70, 50),
        'Timisoara': (50, 200),
        'Lugoj': (150, 200),
        'Mehadia': (200, 250),
        'Drobeta': (250, 300),
        'Craiova': (300, 350),
        'Sibiu': (150, 100),
        'Rimnicu Vilcea': (200, 150),
        'Fagaras': (300, 100),
        'Pitesti': (250, 200),
        'Bucharest': (400, 300),
        'Giurgiu': (400, 400),
        'Urziceni': (450, 350),
        'Hirsova': (500, 400),
        'Eforie': (550, 450),
        'Vaslui': (450, 250),
        'Iasi': (500, 200),
        'Neamt': (550, 150),
    }

    # Dibujar el grafo completo
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=city_positions, with_labels=True, node_size=500, font_size=10, node_color="lightblue", edge_color="gray")

    # Dibujar el camino resaltado en rojo
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos=city_positions, edgelist=path_edges, edge_color='red', width=3)

    # Mostrar las etiquetas con los pesos (distancias)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=city_positions, edge_labels=edge_labels)

    # Mostrar el grafo
    plt.title("Mapa de Rumania - Camino Recorrido")
    plt.show()

# Función para ejecutar la búsqueda y actualizar la interfaz con el resultado
def run_search(algorithm, origin_city, target_city, limit=3):
    if algorithm == 'Búsqueda en Amplitud':
        distances, paths = bfs(romania_map, origin_city)
    elif algorithm == 'DFS':
        distances, paths = dfs(romania_map, origin_city)
    elif algorithm == 'Búsqueda en Profundidad Limitada':
        distances, paths = limited_depth_search(romania_map, origin_city, limit)
    elif algorithm == 'Búsqueda Voraz':
        distances, paths = greedy_search(romania_map, origin_city, target_city, heuristics)
    
    # Mostrar resultado final
    result = f"Distancias desde {origin_city} hasta {target_city} utilizando {algorithm}:\n"
    if target_city in distances:
        result += f"{target_city}: {distances[target_city]} km\n"
        result += f"Ciudades recorridas: {' -> '.join(paths[target_city])}"
        window.after(0, lambda: draw_path_networkx(paths[target_city]))  # Dibujar camino
    else:
        result += f"{target_city} no alcanzada."

    result_label.config(text=result)

# Función para ejecutar la búsqueda en un hilo separado
def run_search_in_thread(algorithm, origin_city, target_city, limit):
    search_thread = threading.Thread(target=run_search, args=(algorithm, origin_city, target_city, limit))
    search_thread.start()

# Crear la interfaz gráfica usando Tkinter
window = tk.Tk()
window.title("Algoritmos de Búsqueda - Mapa de Rumania")

# Frame principal
main_frame = tk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame izquierdo
left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Frame derecho
right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Cargar imagen
map_label = tk.Label(right_frame)
map_label.pack(pady=10)
load_map_image()

# Widgets en left_frame
tk.Label(left_frame, text="Seleccione el algoritmo:").pack()
algorithm_choice = ttk.Combobox(left_frame, values=["Seleccionar", "Búsqueda en Amplitud", "DFS", "Búsqueda en Profundidad Limitada", "Búsqueda Voraz"])
algorithm_choice.pack()
algorithm_choice.current(0)

# Ciudad de origen
tk.Label(left_frame, text="Seleccione la ciudad de origen:").pack()
origin_city_choice = ttk.Combobox(left_frame, values=list(romania_map.keys()))
origin_city_choice.pack()
origin_city_choice.current(0)

# Ciudad de destino
tk.Label(left_frame, text="Seleccione la ciudad destino:").pack()
target_city_choice = ttk.Combobox(left_frame, values=list(romania_map.keys()))
target_city_choice.pack()
target_city_choice.current(0)

# Límite para búsqueda en profundidad limitada
limit_label = tk.Label(left_frame, text="Límite para Búsqueda en Profundidad Limitada:")
limit_label.pack()
limit_entry = tk.Entry(left_frame)
limit_entry.pack()

# Resultado
result_label = tk.Label(left_frame, text="Resultados aparecerán aquí.")
result_label.pack()

# Tabla para la frontera
frontera_table = ttk.Treeview(left_frame, columns=("Pos", "Ciudad"), show="headings", height=5)
frontera_table.heading("Pos", text="Posición")
frontera_table.heading("Ciudad", text="Ciudad")
frontera_table.pack(pady=10)

# Tabla para explorados
explorados_table = ttk.Treeview(left_frame, columns=("Pos", "Ciudad"), show="headings", height=5)
explorados_table.heading("Pos", text="Posición")
explorados_table.heading("Ciudad", text="Ciudad")
explorados_table.pack(pady=10)

# Botón para iniciar búsqueda
search_button = tk.Button(left_frame, text="Iniciar Búsqueda", command=lambda: run_search_in_thread(
    algorithm_choice.get(), origin_city_choice.get(), target_city_choice.get(), int(limit_entry.get() or 3)))
search_button.pack(pady=10)

# Iniciar GUI
window.mainloop()
