import heapq
import time
import csv
from collections import deque

class Puzzle:
    def __init__(self, estado, pai=None, acao=None, custo=0):
        self.estado = estado  # estado atual do tabuleiro (lista de listas)
        self.pai = pai        # nó pai
        self.acao = acao      # ação que levou a este estado
        self.custo = custo    # custo acumulado do caminho
        self.dimensao = len(estado)  # dimensão do tabuleiro (3 para o 8-puzzle)
    
    def __lt__(self, other):
        return self.custo < other.custo
    
    def __eq__(self, other):
        return self.estado == other.estado
    
    def __hash__(self):
        # Converte a matriz em uma tupla de tuplas para torná-la hashable
        return hash(tuple(tuple(row) for row in self.estado))
    
    def encontrar_vazio(self):
        """Encontra a posição do espaço vazio (0) no tabuleiro."""
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                if self.estado[i][j] == 0:
                    return i, j
    
    def eh_objetivo(self, objetivo):
        """Verifica se o estado atual é o estado objetivo."""
        return self.estado == objetivo
    
    def gerar_sucessores(self):
        """Gera todos os estados sucessores possíveis."""
        i, j = self.encontrar_vazio()
        sucessores = []
        
        # Possíveis movimentos: cima, baixo, esquerda, direita
        acoes = [('cima', -1, 0), ('baixo', 1, 0), ('esquerda', 0, -1), ('direita', 0, 1)]
        
        for acao, di, dj in acoes:
            novo_i, novo_j = i + di, j + dj
            
            # Verifica se o movimento é válido
            if 0 <= novo_i < self.dimensao and 0 <= novo_j < self.dimensao:
                # Cria um novo estado movendo o espaço vazio
                novo_estado = [row[:] for row in self.estado]  # cópia profunda
                novo_estado[i][j], novo_estado[novo_i][novo_j] = novo_estado[novo_i][novo_j], novo_estado[i][j]
                
                # Cria um novo nó para o sucessor
                sucessor = Puzzle(novo_estado, self, acao, self.custo + 1)
                sucessores.append(sucessor)
                
        return sucessores
    
    def distancia_manhattan(self, objetivo):

        distancia = 0
        posicoes_alvo = {}
        # Pré-calcula as posições alvo
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
        """Conta quantas peças estão fora do lugar em relação ao objetivo."""
        count = 0
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                if self.estado[i][j] != 0 and self.estado[i][j] != objetivo[i][j]:
                    count += 1
        return count


def reconstruir_caminho(no_final):
    """Reconstrói o caminho da solução a partir do nó final."""
    caminho = []
    atual = no_final
    
    while atual is not None:
        caminho.append(atual)
        atual = atual.pai
    
    return list(reversed(caminho))


def busca_em_largura(estado_inicial, estado_objetivo, tempo_limite=10):
    """Implementa a busca em largura (BFS) com limite de tempo."""
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = deque([inicio_no])
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        # Verifica se excedeu o tempo limite
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
    """Implementa a busca em profundidade (DFS) com limite de profundidade e tempo."""
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = [inicio_no]
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        # Verifica se excedeu o tempo limite
        if time.time() - inicio > tempo_limite:
            return None, nos_expandidos, tempo_limite
            
        no_atual = fronteira.pop()  # Remove o último elemento (LIFO)
        
        if no_atual.custo > limite_profundidade:
            continue
            
        estado_hash = hash(tuple(tuple(row) for row in no_atual.estado))
        if estado_hash not in explorados:
            explorados.add(estado_hash)
            nos_expandidos += 1
            
            if no_atual.eh_objetivo(estado_objetivo):
                fim = time.time()
                return reconstruir_caminho(no_atual), nos_expandidos, fim - inicio
            
            # Adiciona os sucessores à pilha em ordem inversa
            for sucessor in reversed(no_atual.gerar_sucessores()):
                sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
                if sucessor_hash not in explorados:
                    fronteira.append(sucessor)
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def busca_gulosa(estado_inicial, estado_objetivo, heuristica, tempo_limite=10):
    """Implementa a busca gulosa com uma heurística e limite de tempo."""
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    fronteira = [(heuristica(inicio_no, estado_objetivo), inicio_no)]
    heapq.heapify(fronteira)
    explorados = set()
    nos_expandidos = 0
    
    while fronteira:
        # Verifica se excedeu o tempo limite
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
    """Função heurística que calcula a distância de Manhattan."""
    return no.distancia_manhattan(objetivo)


def heuristica_pecas_fora(no, objetivo):
    """Função heurística que conta peças fora do lugar."""
    return no.pecas_fora_do_lugar(objetivo)


def busca_a_estrela(estado_inicial, estado_objetivo, heuristica, tempo_limite=10):
    """Implementa a busca A* com uma heurística e limite de tempo."""
    inicio = time.time()
    
    inicio_no = Puzzle(estado_inicial)
    if inicio_no.eh_objetivo(estado_objetivo):
        return reconstruir_caminho(inicio_no), 1, 0
    
    # A* usa custo do caminho + heurística
    fronteira = [(heuristica(inicio_no, estado_objetivo) + inicio_no.custo, inicio_no)]
    heapq.heapify(fronteira)
    explorados = set()
    nos_expandidos = 0
    
    g_scores = {hash(tuple(tuple(row) for row in inicio_no.estado)): 0}  # Custo do caminho do início até cada nó
    
    while fronteira:
        # Verifica se excedeu o tempo limite
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
            # Calcula novo custo g para este sucessor
            tentative_g = no_atual.custo + 1
            
            sucessor_hash = hash(tuple(tuple(row) for row in sucessor.estado))
            # Se já vimos este estado e o novo caminho não é melhor, ignoramos
            if sucessor_hash in g_scores and tentative_g >= g_scores[sucessor_hash]:
                continue
                
            # Este é o melhor caminho até agora para este sucessor
            g_scores[sucessor_hash] = tentative_g
            f_score = tentative_g + heuristica(sucessor, estado_objetivo)
            heapq.heappush(fronteira, (f_score, sucessor))
    
    fim = time.time()
    return None, nos_expandidos, fim - inicio


def imprimir_caminho(caminho):
    """Imprime o caminho da solução."""
    for i, no in enumerate(caminho):
        print(f"Passo {i}:")
        for linha in no.estado:
            print(linha)
        if i < len(caminho) - 1:
            print(f"Ação: {caminho[i+1].acao}")
        print()


def imprimir_estado(estado):
    """Imprime um estado do tabuleiro."""
    for linha in estado:
        print(linha)


def ler_estados_do_csv(arquivo_csv):
    """Lê os estados iniciais do arquivo CSV."""
    estados = []
    try:
        with open(arquivo_csv, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Pula o cabeçalho
            for row in reader:
                # Converte as strings para inteiros
                numeros = [int(cell) for cell in row]
                # Converte a lista plana para uma matriz 3x3
                estado = [
                    numeros[0:3],
                    numeros[3:6],
                    numeros[6:9]
                ]
                estados.append(estado)
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        # Estado de exemplo em caso de falha na leitura
        estados = [
            [
                [2, 8, 3],
                [1, 6, 4],
                [7, 0, 5]
            ]
        ]
    
    return estados


def comparar_algoritmos(estado_inicial, estado_objetivo, tempo_limite=10):
    """Compara o desempenho dos diferentes algoritmos de busca."""
    print("=" * 80)
    print("Comparando algoritmos de busca para o quebra-cabeça dos 8 números")
    print("=" * 80)
    
    print("/nEstado Inicial:")
    imprimir_estado(estado_inicial)
    print("/nEstado Objetivo:")
    imprimir_estado(estado_objetivo)
    print("/n")
    
    # Tabela de resultados
    print("{:<20} {:<20} {:<20} {:<20}".format(
        "Algoritmo", "Tempo (s)", "Nós Expandidos", "Comprimento do Caminho"
    ))
    print("-" * 80)
    
    # BFS
    caminho, nos, tempo = busca_em_largura(estado_inicial, estado_objetivo, tempo_limite)
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "BFS", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "BFS", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    # DFS
    caminho, nos, tempo = busca_em_profundidade(estado_inicial, estado_objetivo, 30, tempo_limite)
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "DFS", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "DFS", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    # Busca Gulosa (Manhattan)
    caminho, nos, tempo = busca_gulosa(estado_inicial, estado_objetivo, heuristica_manhattan, tempo_limite)
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Manhattan)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Manhattan)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    # Busca Gulosa (Peças fora)
    caminho, nos, tempo = busca_gulosa(estado_inicial, estado_objetivo, heuristica_pecas_fora, tempo_limite)
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Peças Fora)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "Gulosa (Peças Fora)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    # A* (Manhattan)
    caminho, nos, tempo = busca_a_estrela(estado_inicial, estado_objetivo, heuristica_manhattan, tempo_limite)
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
    
    # A* (Peças fora)
    caminho, nos, tempo = busca_a_estrela(estado_inicial, estado_objetivo, heuristica_pecas_fora, tempo_limite)
    if caminho:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Peças Fora)", tempo, nos, len(caminho) - 1
        ))
    else:
        print("{:<20} {:<20.6f} {:<20} {:<20}".format(
            "A* (Peças Fora)", tempo, nos, "Tempo limite excedido" if tempo >= tempo_limite else "Não encontrou"
        ))
    
    print("/nCaminho da melhor solução encontrada:")
    if melhor_caminho:
        for i, no in enumerate([melhor_caminho[0], melhor_caminho[-1]]):
            print(f"{'Estado Inicial' if i == 0 else 'Estado Final'}:")
            imprimir_estado(no.estado)
            print()
        print(f"Número de movimentos: {len(melhor_caminho) - 1}")
    else:
        print("Nenhuma solução encontrada.")
    
    return 1 if melhor_caminho else 0


def avaliar_instancias_csv(arquivo_csv, tempo_limite=30):
    """Avalia todas as instâncias do arquivo CSV."""
    estados = ler_estados_do_csv(arquivo_csv)
    
    # Estado objetivo (configuração ordenada)
    estado_objetivo = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    sucessos = 0
    total = len(estados)
    
    for i, estado in enumerate(estados):
        print(f"/n/nTestando instância {i+1}/{total}")
        sucessos += comparar_algoritmos(estado, estado_objetivo, tempo_limite)
    
    print(f"/n/nResultados Finais: {sucessos}/{total} instâncias resolvidas com sucesso.")


# Exemplo de uso
if __name__ == "__main__":
    arquivo_csv = "H:/Meu Drive/Ciencia da Computação/05P/INTELIGÊNCIA ARTIFICIAL/1BIM/ED2/ed02-puzzle8.csv"
    avaliar_instancias_csv(arquivo_csv, tempo_limite=30)  # 30 segundos por instância