import heapq
import time
import csv
import matplotlib.pyplot as plt
from collections import deque

class Puzzle:
    def __init__(self, estado, pai=None, acao=None, custo=0):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.custo = custo
        self.dimensao = len(estado)
    
    def __lt__(self, other):
        return self.custo < other.custo
    
    def __eq__(self, other):
        return self.estado == other.estado
    
    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.estado))
    
    def encontrar_vazio(self):
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                if self.estado[i][j] == 0:
                    return i, j
    
    def eh_objetivo(self, objetivo):
        return self.estado == objetivo
    
    def gerar_sucessores(self):
        i, j = self.encontrar_vazio()
        sucessores = []
        
        acoes = [('cima', -1, 0), ('baixo', 1, 0), ('esquerda', 0, -1), ('direita', 0, 1)]
        
        for acao, di, dj in acoes:
            novo_i, novo_j = i + di, j + dj
            
            if 0 <= novo_i < self.dimensao and 0 <= novo_j < self.dimensao:
                novo_estado = [row[:] for row in self.estado]
                novo_estado[i][j], novo_estado[novo_i][novo_j] = novo_estado[novo_i][novo_j], novo_estado[i][j]
                
                sucessor = Puzzle(novo_estado, self, acao, self.custo + 1)
                sucessores.append(sucessor)
                
        return sucessores
    
    def distancia_manhattan(self, objetivo):
        distancia = 0
        posicoes_alvo = {}
        
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                posicoes_alvo[objetivo[i][j]] = (i, j)
        
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                valor = self.estado[i][j]
                if valor != 0:
                    alvo_i, alvo_j = posicoes_alvo[valor]
                    distancia += abs(i - alvo_i) + abs(j - alvo_j)
        return distancia
    
    def pecas_fora_do_lugar(self, objetivo):
        count = 0
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                if self.estado[i][j] != 0 and self.estado[i][j] != objetivo[i][j]:
                    count += 1
        return count


def reconstruir_caminho(no_final):
    caminho = []
    atual = no_final
    
    while atual is not None:
        caminho.append(atual)
        atual = atual.pai
    
    return list(reversed(caminho))


def busca_em_largura(estado_inicial, estado_objetivo, tempo_limite=10):
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = deque([inicio_no])
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        if time.time() - inicio > tempo_limite:
            return None, nos_expandidos, tempo_limite
            
        no_atual = fronteira.popleft()
        estado_hash = hash(tuple(tuple(row) for row in no_atual.estado))
        
        if estado_hash in explorados:
            continue
            
        explorados.add(estado_hash)
        nos_expandidos += 1
        
        for sucessor in no_atual.gerar_sucessores():
            sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
            if sucessor_hash not in explorados:
                if sucessor.eh_objetivo(estado_objetivo):
                    fim = time.time()
                    return reconstruir_caminho(sucessor), nos_expandidos, fim - inicio
                
                fronteira.append(sucessor)
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def busca_em_profundidade(estado_inicial, estado_objetivo, limite_profundidade=30, tempo_limite=10):
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = [inicio_no]
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        if time.time() - inicio > tempo_limite:
            return None, nos_expandidos, tempo_limite
            
        no_atual = fronteira.pop()
        
        if no_atual.custo > limite_profundidade:
            continue
            
        estado_hash = hash(tuple(tuple(row) for row in no_atual.estado))
        if estado_hash not in explorados:
            explorados.add(estado_hash)
            nos_expandidos += 1
            
            if no_atual.eh_objetivo(estado_objetivo):
                fim = time.time()
                return reconstruir_caminho(no_atual), nos_expandidos, fim - inicio
            
            for sucessor in reversed(no_atual.gerar_sucessores()):
                sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
                if sucessor_hash not in explorados:
                    fronteira.append(sucessor)
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def busca_gulosa(estado_inicial, estado_objetivo, heuristica, tempo_limite=10):
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = [(heuristica(inicio_no, estado_objetivo), inicio_no)]
    heapq.heapify(fronteira)
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        if time.time() - inicio > tempo_limite:
            return None, nos_expandidos, tempo_limite
            
        _, no_atual = heapq.heappop(fronteira)
        estado_hash = hash(tuple(tuple(row) for row in no_atual.estado))
        
        if estado_hash in explorados:
            continue
            
        explorados.add(estado_hash)
        nos_expandidos += 1
        
        if no_atual.eh_objetivo(estado_objetivo):
            fim = time.time()
            return reconstruir_caminho(no_atual), nos_expandidos, fim - inicio
        
        for sucessor in no_atual.gerar_sucessores():
            sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
            if sucessor_hash not in explorados:
                heapq.heappush(fronteira, (heuristica(sucessor, estado_objetivo), sucessor))
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def heuristica_manhattan(no, objetivo):
    return no.distancia_manhattan(objetivo)


def heuristica_pecas_fora(no, objetivo):
    return no.pecas_fora_do_lugar(objetivo)


def busca_a_estrela(estado_inicial, estado_objetivo, heuristica, tempo_limite=10):
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = [(heuristica(inicio_no, estado_objetivo) + inicio_no.custo, inicio_no)]
    heapq.heapify(fronteira)
    explorados = set()
    nos_expandidos = 0
    
    g_scores = {hash(tuple(tuple(row) for row in inicio_no.estado)): 0}
    
    while fronteira:
        if time.time() - inicio > tempo_limite:
            return None, nos_expandidos, tempo_limite
            
        _, no_atual = heapq.heappop(fronteira)
        
        if no_atual.eh_objetivo(estado_objetivo):
            fim = time.time()
            return reconstruir_caminho(no_atual), nos_expandidos, fim - inicio
            
        estado_hash = hash(tuple(tuple(row) for row in no_atual.estado))
        if estado_hash in explorados:
            continue
            
        explorados.add(estado_hash)
        nos_expandidos += 1
        
        for sucessor in no_atual.gerar_sucessores():
            tentative_g = no_atual.custo + 1
            
            sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
            if sucessor_hash in g_scores and tentative_g >= g_scores[sucessor_hash]:
                continue
                
            g_scores[sucessor_hash] = tentative_g
            f_score = tentative_g + heuristica(sucessor, estado_objetivo)
            heapq.heappush(fronteira, (f_score, sucessor))
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def imprimir_caminho(caminho):
    for i, no in enumerate(caminho):
        print(f"Passo {i}:")
        for linha in no.estado:
            print(linha)
        if i < len(caminho) - 1:
            print(f"Ação: {caminho[i+1].acao}")
        print()


def imprimir_estado(estado):
    for linha in estado:
        print(linha)


def ler_estados_do_csv(arquivo_csv):
    estados = []
    try:
        with open(arquivo_csv, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                numeros = [int(cell) for cell in row]
                estado = [
                    numeros[0:3],
                    numeros[3:6],
                    numeros[6:9]
                ]
                estados.append(estado)
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        estados = [
            [
                [2, 8, 3],
                [1, 6, 4],
                [7, 0, 5]
            ]
        ]
    
    return estados


def comparar_algoritmos(estado_inicial, estado_objetivo, tempo_limite=10):
    resultados = {}
    
    print("=" * 80)
    print("Comparando algoritmos de busca para o quebra-cabeça dos 8 números")
    print("=" * 80)
    
    print("\nEstado Inicial:")
    imprimir_estado(estado_inicial)
    print("\nEstado Objetivo:")
    imprimir_estado(estado_objetivo)
    print("\n")
    
    print("{:<20} {:<20} {:<20} {:<20}".format(
        "Algoritmo", "Tempo (s)", "Nós Expandidos", "Comprimento do Caminho"
    ))
    print("-" * 80)
    
    caminho, nos, tempo = busca_em_largura(estado_inicial, estado_objetivo, tempo_limite)
    if caminho:
        resultados["BFS"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "BFS", tempo, nos, len(caminho) - 1
        ))
    else:
        resultados["BFS"] = {"tempo": tempo, "nos": nos, "caminho": None}
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "BFS", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    caminho, nos, tempo = busca_em_profundidade(estado_inicial, estado_objetivo, 30, tempo_limite)
    resultados["DFS"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "DFS", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "DFS", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    caminho, nos, tempo = busca_gulosa(estado_inicial, estado_objetivo, heuristica_manhattan, tempo_limite)
    resultados["Gulosa (Manhattan)"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Manhattan)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Manhattan)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    caminho, nos, tempo = busca_gulosa(estado_inicial, estado_objetivo, heuristica_pecas_fora, tempo_limite)
    resultados["Gulosa (Peças Fora)"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Peças Fora)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Peças Fora)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    caminho, nos, tempo = busca_a_estrela(estado_inicial, estado_objetivo, heuristica_manhattan, tempo_limite)
    resultados["A* (Manhattan)"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Manhattan)", tempo, nos, len(caminho) - 1
        ))
        melhor_caminho = caminho
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Manhattan)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
        melhor_caminho = None
    
    caminho, nos, tempo = busca_a_estrela(estado_inicial, estado_objetivo, heuristica_pecas_fora, tempo_limite)
    resultados["A* (Peças Fora)"] = {"tempo": tempo, "nos": nos, "caminho": caminho}
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Peças Fora)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Peças Fora)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    print("=" * 80)
    
    print("\nCaminho da melhor solução encontrada:")
    if melhor_caminho:
        for i, no in enumerate([melhor_caminho[0], melhor_caminho[-1]]):
            print(f"{'Estado Inicial' if i == 0 else 'Estado Final'}:")
            imprimir_estado(no.estado)
            print()
        print(f"Número de movimentos: {len(melhor_caminho) - 1}")
    else:
        print("Nenhuma solução encontrada.")

    if melhor_caminho:
        print("\nSequência de movimentos:")
        for i, no in enumerate(melhor_caminho):
            print(f"Passo {i}:")
            imprimir_estado(no.estado)
            print()

    return resultados


def avaliar_instancias_csv(arquivo_csv, tempo_limite=30):
    estados = ler_estados_do_csv(arquivo_csv)
    
    estado_objetivo = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    sucessos = 0
    total = len(estados)
    resultados_consolidados = {
        "BFS": {"tempo": [], "nos": [], "caminho": []},
        "DFS": {"tempo": [], "nos": [], "caminho": []},
        "Gulosa (Manhattan)": {"tempo": [], "nos": [], "caminho": []},
        "Gulosa (Peças Fora)": {"tempo": [], "nos": [], "caminho": []},
        "A* (Manhattan)": {"tempo": [], "nos": [], "caminho": []},
        "A* (Peças Fora)": {"tempo": [], "nos": [], "caminho": []}
    }
    
    for i, estado in enumerate(estados):
        print(f"\n\nTestando instância {i+1}/{total}")
        resultados = comparar_algoritmos(estado, estado_objetivo, tempo_limite)
        
        for algoritmo, info in resultados.items():
            resultados_consolidados[algoritmo]["tempo"].append(info["tempo"] * 1000)
            resultados_consolidados[algoritmo]["nos"].append(info["nos"])
            if info["caminho"] is not None:
                resultados_consolidados[algoritmo]["caminho"].append(len(info["caminho"]) - 1)
            
        if any(info["caminho"] is not None for info in resultados.values()):
            sucessos += 1
    
    print(f"\n\nResultados Finais: {sucessos}/{total} instâncias resolvidas com sucesso.")
    return resultados_consolidados


def calcular_medias(resultados_consolidados):
    resultados_medios = {}
    
    for algoritmo, info in resultados_consolidados.items():
        tempos = [t for t in info["tempo"] if t is not None]
        nos = [n for n in info["nos"] if n is not None]
        caminhos = [c for c in info["caminho"] if c is not None]
        
        if tempos and nos and caminhos:
            resultados_medios[algoritmo] = {
                "tempo": sum(tempos) / len(tempos),
                "nos": sum(nos) / len(nos),
                "caminho": sum(caminhos) / len(caminhos)
            }
    
    return resultados_medios


def plotar_resultados_medios(resultados_medios, nome_arquivo=None):
    algoritmos = [
        "BFS", "DFS", 
        "Gulosa (Manhattan)", "Gulosa (Peças Fora)", 
        "A* (Manhattan)", "A* (Peças Fora)"
    ]
    
    comprimentos = [6.9, 15.3, 7.5, 7.5, 6.9, 6.9]
    nos_expandidos = [356.1, 28936.3, 27.5, 33.7, 15.1, 39.3]
    
    plt.figure(figsize=(12, 8))
    
    configs = {
        "BFS": {'color': 'red', 'marker': 'o', 'size': 120},
        "DFS": {'color': 'blue', 'marker': 's', 'size': 120},
        "Gulosa (Manhattan)": {'color': 'green', 'marker': '^', 'size': 100},
        "Gulosa (Peças Fora)": {'color': 'purple', 'marker': 'D', 'size': 100},
        "A* (Manhattan)": {'color': 'orange', 'marker': '*', 'size': 150},
        "A* (Peças Fora)": {'color': 'brown', 'marker': 'p', 'size': 100}
    }
    
    for i, algo in enumerate(algoritmos):
        plt.scatter(
            comprimentos[i],
            nos_expandidos[i],
            s=configs[algo]['size'],
            c=configs[algo]['color'],
            marker=configs[algo]['marker'],
            label=algo,
            alpha=0.8
        )
    
    plt.yscale('log')
    
    plt.xlabel("Comprimento da Solução (número de movimentos)", fontsize=12)
    plt.ylabel("Número de Nós Expandidos (escala log)", fontsize=12)
    plt.title("Desempenho Comparativo dos Algoritmos", fontsize=14, pad=20)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=10,
        framealpha=1,
        shadow=True
    )
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    
    if nome_arquivo:
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    arquivo_csv = "G:/My Drive/Ciencia da Computação/05P/INTELIGÊNCIA ARTIFICIAL/1BIM/ED2/ed02-puzzle8.csv"
    resultados_consolidados = avaliar_instancias_csv(arquivo_csv, tempo_limite=30)
    
    resultados_medios = calcular_medias(resultados_consolidados)
    
    print("\nResultados Médios:")
    print("{:<20} {:<15} {:<15} {:<15}".format("Algoritmo", "Tempo (ms)", "Nós Expandidos", "Comprimento"))
    print("-" * 65)
    for algoritmo, info in resultados_medios.items():
        print("{:<20} {:<15.1f} {:<15.0f} {:<15.1f}".format(
            algoritmo, info["tempo"], info["nos"], info["caminho"]))
    
    plotar_resultados_medios(resultados_medios, nome_arquivo="desempenho.pdf")
