import pandas as pd
import numpy as np
import io
import time
import os

# =============================================================================
# 0. Configurações e Lista de Arquivos
# =============================================================================

# Lista dos arquivos do problema da mochila a serem processados
knapsack_files = [f'knapsack_{i}.csv' for i in range(1, 11)]

# =============================================================================
# 1. Funções do Algoritmo Genético
# =============================================================================

def calculate_fitness(chromosome, weights, values, capacity):
    """
    Calcula a aptidão (fitness) de um cromossomo.
    A aptidão é o valor total dos itens selecionados.
    Soluções que excedem a capacidade da mochila recebem aptidão 0.
    """
    total_weight = np.sum(chromosome * weights)
    total_value = np.sum(chromosome * values)
    if total_weight > capacity:
        return 0
    else:
        return total_value

def initialize_population_random(pop_size, num_items):
    """
    Inicializa uma população de cromossomos aleatórios (0s e 1s).
    """
    return np.random.randint(0, 2, size=(pop_size, num_items))

def initialize_population_heuristic(pop_size, num_items, weights, values, capacity):
    """
    Inicializa parte da população usando uma heurística gulosa (baseada na razão valor/peso)
    e o restante aleatoriamente para garantir diversidade.
    """
    population = []
    
    value_per_weight = np.divide(values, weights, out=np.zeros_like(values, dtype=float), where=weights!=0)
    sorted_indices = np.argsort(value_per_weight)[::-1]

    num_heuristic = pop_size // 2
    if num_heuristic == 0 and pop_size > 0: num_heuristic = 1

    for _ in range(num_heuristic):
        chromosome = np.zeros(num_items, dtype=int)
        current_weight = 0
        
        for i in sorted_indices:
            if current_weight + weights[i] <= capacity:
                chromosome[i] = 1
                current_weight += weights[i]
        
        population.append(chromosome)

    while len(population) < pop_size:
        population.append(np.random.randint(0, 2, size=num_items))
        
    return np.array(population)

def roulette_wheel_selection(population, weights, values, capacity, num_parents):
    """
    Realiza a seleção de pais utilizando o método da Roleta.
    A probabilidade de seleção de um indivíduo é proporcional à sua aptidão.
    """
    fitness_scores = np.array([calculate_fitness(c, weights, values, capacity) for c in population])
    
    total_fitness = np.sum(fitness_scores)
    if total_fitness == 0:
        return population[np.random.choice(len(population), size=num_parents, replace=True)]
    
    probabilities = fitness_scores / total_fitness
    cumulative_probabilities = np.cumsum(probabilities)

    selected_parents = []
    for _ in range(num_parents):
        r = np.random.rand()
        for i, cum_prob in enumerate(cumulative_probabilities):
            if r <= cum_prob:
                selected_parents.append(population[i])
                break
    return np.array(selected_parents)

def tournament_selection(population, weights, values, capacity, num_parents, tournament_size):
    """
    Realiza a seleção de pais utilizando o método do Torneio.
    Indivíduos são selecionados aleatoriamente para um torneio, e o melhor é escolhido.
    """
    selected_parents = []
    pop_size = len(population)
    for _ in range(num_parents):
        tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
        tournament_competitors = population[tournament_indices]

        best_fitness = -1
        best_competitor = None
        for competitor in tournament_competitors:
            current_fitness = calculate_fitness(competitor, weights, values, capacity)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_competitor = competitor
        selected_parents.append(best_competitor)
    return np.array(selected_parents)

def one_point_crossover(parent1, parent2):
    """
    Realiza o crossover de um ponto entre dois cromossomos.
    """
    chromosome_length = len(parent1)
    if chromosome_length <= 1:
        return np.copy(parent1), np.copy(parent2)
    
    crossover_point = np.random.randint(1, chromosome_length)
    
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def two_point_crossover(parent1, parent2):
    """
    Realiza o crossover de dois pontos entre dois cromossomos.
    """
    chromosome_length = len(parent1)
    if chromosome_length <= 2:
        return np.copy(parent1), np.copy(parent2)
    
    points = sorted(np.random.choice(range(1, chromosome_length), size=2, replace=False))
    point1, point2 = points

    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return child1, child2

def uniform_crossover(parent1, parent2):
    """
    Realiza o crossover uniforme entre dois cromossomos.
    Cada gene é trocado entre os pais com 50% de probabilidade.
    """
    chromosome_length = len(parent1)
    child1 = np.zeros(chromosome_length, dtype=int)
    child2 = np.zeros(chromosome_length, dtype=int)
    for i in range(chromosome_length):
        if np.random.rand() < 0.5:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    return child1, child2

def mutate(chromosome, mutation_rate):
    """
    Aplica a operação de mutação (bit-flip) a um cromossomo.
    Inverte um bit com uma certa probabilidade (taxa de mutação).
    """
    mutated_chromosome = np.copy(chromosome)
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome

# =============================================================================
# 2. Função Principal do Algoritmo Genético
# =============================================================================

def genetic_algorithm_run(
    pop_size,
    num_generations,
    crossover_rate,
    mutation_rate,
    selection_type,
    crossover_type,
    initialization_type,
    stopping_criterion,
    weights,
    values,
    knapsack_capacity,
    num_items,
    tournament_size=5,
    convergence_threshold_generations=20
):
    """
    Executa o Algoritmo Genético para um problema da mochila específico.
    Retorna a melhor aptidão encontrada e o número de gerações executadas.
    """
    if initialization_type == 'random':
        population = initialize_population_random(pop_size, num_items)
    elif initialization_type == 'heuristic':
        population = initialize_population_heuristic(pop_size, num_items, weights, values, knapsack_capacity)
    else:
        raise ValueError("Tipo de inicialização inválido. Use 'random' ou 'heuristic'.")

    best_fitness_overall = -1
    best_solution_overall = None
    generations_without_improvement = 0
    generations_executed = 0

    for generation in range(num_generations):
        generations_executed += 1
        fitness_scores = np.array([calculate_fitness(c, weights, values, knapsack_capacity) for c in population])

        current_best_fitness = np.max(fitness_scores)
        
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_solution_overall = population[np.argmax(fitness_scores)]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if stopping_criterion == 'convergence' and generations_without_improvement >= convergence_threshold_generations:
            break
        
        if selection_type == 'roulette':
            parents = roulette_wheel_selection(population, weights, values, knapsack_capacity, pop_size)
        elif selection_type == 'tournament':
            parents = tournament_selection(population, weights, values, knapsack_capacity, pop_size, tournament_size)
        else:
            raise ValueError("Tipo de seleção inválido. Use 'roulette' ou 'tournament'.")

        next_population = []
        
        if best_solution_overall is not None and calculate_fitness(best_solution_overall, weights, values, knapsack_capacity) > 0:
             next_population.append(np.copy(best_solution_overall))
        else:
            next_population.append(population[np.random.randint(pop_size)])

        num_offspring_needed = pop_size - len(next_population)
        num_pairs = num_offspring_needed // 2

        for i in range(num_pairs):
            p1_idx, p2_idx = np.random.choice(len(parents), size=2, replace=False)
            parent1 = parents[p1_idx]
            parent2 = parents[p2_idx]

            if np.random.rand() < crossover_rate:
                if crossover_type == 'one_point':
                    child1, child2 = one_point_crossover(parent1, parent2)
                elif crossover_type == 'two_point':
                    child1, child2 = two_point_crossover(parent1, parent2)
                elif crossover_type == 'uniform':
                    child1, child2 = uniform_crossover(parent1, parent2)
                else:
                    raise ValueError("Tipo de crossover inválido.")
            else:
                child1, child2 = np.copy(parent1), np.copy(parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            next_population.extend([child1, child2])
        
        if len(next_population) < pop_size:
            next_population.append(mutate(parents[np.random.choice(len(parents))], mutation_rate))

        population = np.array(next_population[:pop_size])

    # Corrigido: Retorna apenas 2 valores como esperado por run_comparison
    return best_fitness_overall, generations_executed


# =============================================================================
# 3. Função para Análise de Comparações
# =============================================================================

def run_comparison(title, param_configs, common_params, weights, values, knapsack_capacity, num_items, num_runs=10):
    """
    Executa a comparação de parâmetros para um problema da mochila específico,
    rodando o AG múltiplas vezes para cada configuração e coletando estatísticas.
    """
    results = {}
    for config_name, specific_params in param_configs.items():
        run_best_fitnesses = []
        run_generations = []
        
        for run in range(num_runs):
            params = {**common_params, **specific_params}
            # Corrigido: Agora espera 2 valores, não 4
            best_fitness, generations_executed = genetic_algorithm_run(
                pop_size=params['pop_size'],
                num_generations=params['num_generations'],
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate'],
                selection_type=params['selection_type'],
                crossover_type=params['crossover_type'],
                initialization_type=params['initialization_type'],
                stopping_criterion=params['stopping_criterion'],
                weights=weights,
                values=values,
                knapsack_capacity=knapsack_capacity,
                num_items=num_items,
                tournament_size=params.get('tournament_size', 5),
                convergence_threshold_generations=params.get('convergence_threshold_generations', 20)
            )
            run_best_fitnesses.append(best_fitness)
            run_generations.append(generations_executed)
        
        avg_best_fitness = np.mean(run_best_fitnesses)
        std_best_fitness = np.std(run_best_fitnesses)
        avg_generations = np.mean(run_generations)
        results[config_name] = {
            'avg_best_fitness': avg_best_fitness,
            'std_best_fitness': std_best_fitness,
            'avg_generations': avg_generations
        }
    return results


# =============================================================================
# 4. Loop Principal para Processar Todos os Arquivos e Gerar Relatório
# =============================================================================

# Dicionário para armazenar todos os resultados para o relatório final
all_problems_results = {}

# Configurações Base para as Comparações
base_params = {
    'pop_size': 50,
    'num_generations': 100,
    'crossover_rate': 0.8,
    'mutation_rate': 0.05,
    'selection_type': 'tournament',
    'crossover_type': 'one_point',
    'initialization_type': 'random',
    'stopping_criterion': 'fixed_generations',
    'tournament_size': 5,
    'convergence_threshold_generations': 20
}

# Definições das configurações específicas para cada tipo de comparação
mutation_configs = {
    'Mutação Baixa (0.01)': {'mutation_rate': 0.01},
    'Mutação Média (0.05)': {'mutation_rate': 0.05},
    'Mutação Alta (0.10)': {'mutation_rate': 0.10}
}
crossover_configs = {
    'Crossover Um Ponto': {'crossover_type': 'one_point'},
    'Crossover Dois Pontos': {'crossover_type': 'two_point'},
    'Crossover Uniforme': {'crossover_type': 'uniform'}
}
selection_configs = {
    'Seleção por Roleta': {'selection_type': 'roulette'},
    'Seleção por Torneio': {'selection_type': 'tournament'}
}
initialization_configs = {
    'Inicialização Aleatória': {'initialization_type': 'random'},
    'Inicialização Heurística': {'initialization_type': 'heuristic'}
}
stopping_configs = {
    'Critério: Fixo (100 Gens)': {'stopping_criterion': 'fixed_generations', 'num_generations': 100},
    'Critério: Convergência (20 Gens sem melhora)': {'stopping_criterion': 'convergence', 'num_generations': 200}
}

# Prepara a string do relatório para impressão e salvamento em arquivo
report_output = io.StringIO()
report_output.write("="*80 + "\n")
report_output.write("RELATÓRIO DE ANÁLISE COMPARATIVA DO ALGORITMO GENÉTICO PARA PROBLEMAS DA MOCHILA\n")
report_output.write("="*80 + "\n\n")

for filename in knapsack_files:
    report_output.write(f"\n### Processando Arquivo: {filename} ###\n")
    print(f"\n--- Processando Arquivo: {filename} ---") # Andamento no terminal

    try:
        if not os.path.exists(filename):
            report_output.write(f"ERRO: O arquivo '{filename}' não foi encontrado na mesma pasta do script.\n")
            report_output.write("Por favor, certifique-se de que todos os arquivos CSV estão presentes.\n\n")
            print(f"ERRO: O arquivo '{filename}' não foi encontrado. Pulando para o próximo.")
            continue

        current_df = pd.read_csv(filename)

        current_knapsack_capacity = current_df.iloc[-1]['Peso']
        current_items_df = current_df.iloc[:-1]
        current_weights = current_items_df['Peso'].values
        current_values = current_items_df['Valor'].values
        current_num_items = len(current_weights)

        report_output.write(f"  Capacidade da Mochila: {current_knapsack_capacity}\n")
        report_output.write(f"  Número de Itens: {current_num_items}\n")
        report_output.write(f"  Pesos dos Itens: {current_weights}\n")
        report_output.write(f"  Valores dos Itens: {current_values}\n\n")

        problem_results_for_file = {}
        
        all_comparisons = {
            "Taxas de Mutação": (mutation_configs, base_params),
            "Tipos de Crossover": (crossover_configs, base_params),
            "Tipos de Seleção": (selection_configs, base_params),
            "Inicialização da População": (initialization_configs, base_params),
            "Critério de Parada": (stopping_configs, base_params)
        }
        
        for category_title, (configs, common) in all_comparisons.items():
            print(f"  > Executando comparações para: {category_title}...") # Andamento no terminal
            report_output.write(f"--- {category_title} para {filename} ---\n")
            results = run_comparison(category_title, configs, common, current_weights, current_values, current_knapsack_capacity, current_num_items)
            problem_results_for_file[category_title] = results

            # Escrever resultados para o relatório de texto
            for config_name, data in results.items():
                report_output.write(f"    Configuração: {config_name}:\n")
                report_output.write(f"      Melhor Aptidão Média: {data['avg_best_fitness']:.2f} (Desvio Padrão: {data['std_best_fitness']:.2f})\n")
                report_output.write(f"      Gerações Médias Executadas: {data['avg_generations']:.1f}\n")
            report_output.write("\n")

        report_output.write("\n" + "="*70 + "\n\n") # Separador para cada arquivo
        all_problems_results[filename] = problem_results_for_file

    except Exception as e:
        report_output.write(f"ERRO ao processar '{filename}': {e}\n\n")
        print(f"ERRO ao processar '{filename}': {e}. Pulando para o próximo.")
        continue


# =============================================================================
# 5. Saída do Relatório Final (apenas em arquivo)
# =============================================================================

final_report_string = report_output.getvalue()

# Salvar o relatório em um arquivo de texto
report_filename = "relatorio_ag_mochila.txt"
with open(report_filename, "w") as f:
    f.write(final_report_string)

print(f"\nRelatório completo salvo em '{report_filename}'")
print("\nAnálise concluída!")