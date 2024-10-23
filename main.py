from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from scipy.optimize import linear_sum_assignment
import uvicorn

app = FastAPI()

class EdgeList(BaseModel):
    edges: List[Tuple[str, str, Optional[float]]]  # Adicionando peso opcional às arestas
    nodes: List[str] = []  # Lista de nós isolados
    directed: bool = True  # Opção para grafo orientado ou não

class GraphInput(BaseModel):
    graph: dict

class Node(BaseModel):
    node: str
    directed: bool = True


def cria_grafo(orientado=True):
    """Cria um grafo (orientado ou não)"""
    return nx.DiGraph() if orientado else nx.Graph()


def adicionar_arestas(G, arestas):
    """Adiciona uma lista de arestas ao grafo"""
    G.add_weighted_edges_from(arestas)


def adicionar_nos(G, nos):
    """Adiciona uma lista de nós isolados ao grafo"""
    G.add_nodes_from(nos)


def desenhar_grafo(G):
    """Desenha o grafo e retorna a imagem como bytes"""
    pos = nx.spring_layout(G, k=0.1)  # layout para os nós do grafo, ajustando o parâmetro k para aproximar os nós
    plt.figure(figsize=(6, 4))  # define o tamanho da figura
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue",
            font_size=10, font_color="black", edge_color="gray")
    plt.title("Visualização do Grafo")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
    

def hungarian_algorithm(json_input):
    # Extrai coluna e linha
    rows = list(json_input.keys())
    columns = list(json_input[rows[0]].keys())

    cost_matrix = np.array([[json_input[row][col] for col in columns] for row in rows])

    # Se a matriz nao for quadrada, adicione 0's onde for necessário
    num_rows, num_cols = cost_matrix.shape
    if num_rows < num_cols:
        # Adiciona 0 extras
        padded_cost_matrix = np.zeros((num_cols, num_cols))
        padded_cost_matrix[:num_rows, :] = cost_matrix
    else:
        padded_cost_matrix = cost_matrix

    # Aplica o algoritmo
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)

    # Extrai os resultados
    result = []
    for i, j in zip(row_ind, col_ind):
        if i < num_rows:  # Somente vai incluir os resultados de vertices existentes
            result.append(((rows[i], columns[j]), json_input[rows[i]][columns[j]]))

    return result


@app.post("/grafo/imagem", summary="Desenha e retorna o grafo como uma imagem", response_description="Imagem do grafo")
async def criar_e_desenhar_grafo(data: EdgeList):
    """
    Recebe uma lista de arestas e nós, e a opção de ser orientado ou não, cria e desenha o grafo.

    - **edges**: Lista de arestas onde cada aresta é um par de inteiros.
    - **nodes**: Lista de nós isolados.
    - **directed**: Booleano indicando se o grafo é orientado (padrão: True).
    """
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    image_buf = desenhar_grafo(G)
    return StreamingResponse(image_buf, media_type="image/png")


@app.post("/grafo/arvore_geradora_minima", summary="Encontra a árvore geradora mínima de um grafo ponderado",
          response_description="Árvore geradora mínima")
async def arvore_geradora_minima(data: EdgeList):
    if data.directed:
        return "A árvore geradora mínima só é aplicável a grafos não-orientados."
    G = cria_grafo(orientado=False)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    mst = nx.minimum_spanning_tree(G)
    edges = list(mst.edges(data=True))

    edge_list_str = ", ".join([f"({u}, {v}, peso: {d['weight']})" for u, v, d in edges])
    return f"A árvore geradora mínima contém as seguintes arestas: {edge_list_str}."


@app.post("/grafo/algoritmo_hungaro", summary="Implementa o Algoritmo Húngaro em um grafo bipartido completo ponderado",
          response_description="Emparelhamento mínimo de custo")
async def algoritmo_hungaro(data: EdgeList):
    G = cria_grafo(orientado=False)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    if not nx.is_bipartite(G):
        return "O Algoritmo Húngaro só é aplicável a grafos bipartidos."

    mate = nx.algorithms.bipartite.minimum_weight_full_matching(G)
    matching_str = ", ".join([f"({u}, {v})" for u, v in mate.items()])

    return f"O emparelhamento mínimo de custo é: {matching_str}."


@app.post("/grafo/verificar_existencia_aresta", summary="Verifica a existência de uma aresta",
          response_description="Existência da aresta")
async def verificar_existencia_aresta(data: EdgeList, u: str, v: str):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    existencia = G.has_edge(u, v)
    return f"A aresta entre {u} e {v} {'existe' if existencia else 'não existe'}."


@app.post("/grafo/grau_vertice", summary="Retorna o grau de um vértice", response_description="Grau do vértice")
async def grau_vertice(data: EdgeList, v: str):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    grau = G.degree[v]
    return f"O grau do vértice {v} é {grau}."


@app.post("/grafo/adjacencia_vertice", summary="Retorna a adjacência de um vértice",
          response_description="Lista de adjacências")
async def adjacencia_vertice(data: EdgeList, v: str):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    adjacencia = list(G.adj[v])
    return f"A adjacência do vértice {v} é {adjacencia}."


@app.post("/grafo/verificar_grafo_ciclico", summary="Verifica se o grafo contém ciclos",
          response_description="Se o grafo é cíclico")
async def verificar_grafo_ciclico(data: EdgeList):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    ciclico = not nx.is_directed_acyclic_graph(G) if isinstance(G, nx.DiGraph) else len(list(nx.cycle_basis(G))) > 0
    return f"O grafo {'é' if ciclico else 'não é'} cíclico."


@app.post("/grafo/componentes_fortemente_conexos", summary="Retorna os componentes fortemente conexos e a quantidade",
          response_description="Componentes fortemente conexos e a quantidade")
async def componentes_fortemente_conexos(data: EdgeList):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    
    if isinstance(G, nx.DiGraph):
        # Obtém os componentes fortemente conectados
        componentes = [list(c) for c in nx.strongly_connected_components(G)]
        # Conta o número de componentes fortemente conectados
        numero_componentes = len(componentes)
        
        return {
            f"O grafo possui {numero_componentes} componente(s) fortemente conectado(s), sendo eles os seguintes: {componentes}"
        }
    else:
        return "O grafo não possui elementos fortemente conectados."



@app.post("/grafo/ordenacao_topologica", summary="Retorna a ordenação topológica",
          response_description="Ordenação topológica")
async def ordenacao_topologica(data: EdgeList):
    print(data)
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    if nx.is_directed_acyclic_graph(G):
        ordenacao = list(nx.topological_sort(G))
        return f"A ordenação topológica do grafo é: {ordenacao}."
    else:
        return "O grafo não é um dígrafo acíclico."


@app.post("/grafo/verificar_euleriano", summary="Verifica se o grafo é euleriano",
          response_description="Se o grafo é euleriano")
async def verificar_euleriano(data: EdgeList):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    euleriano = nx.is_eulerian(G)
    circuito = list(nx.eulerian_circuit(G)) if euleriano else None
    return f"O grafo {'é' if euleriano else 'não é'} euleriano. {'Circuito euleriano: ' + str(circuito) if circuito else ''}"


@app.post("/grafo/verificar_conjunto_independente", summary="Verifica se um conjunto de vértices é independente",
          response_description="Se o conjunto é independente")
async def verificar_conjunto_independente(data: EdgeList, conjunto: List[str]):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    subgraph = G.subgraph(conjunto)
    independente = not subgraph.edges()
    return f"O conjunto {conjunto} {'é' if independente else 'não é'} independente."


@app.post("/grafo/verificar_clique", summary="Verifica se um conjunto de vértices é um clique",
          response_description="Se o conjunto é um clique")
async def verificar_clique(data: EdgeList, conjunto: List[str]):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    subgraph = G.subgraph(conjunto)
    n = len(conjunto)
    clique = len(subgraph.edges()) == n * (n - 1) // 2
    return f"O conjunto {conjunto} {'é' if clique else 'não é'} um clique."


@app.post("/grafo/verificar_conjunto_dominante", summary="Verifica se um conjunto de vértices é dominante",
          response_description="Se o conjunto é dominante")
async def verificar_conjunto_dominante(data: EdgeList, conjunto: List[str]):
    # Verifica se o grafo é orientado, pois a verificação de conjunto dominante é apenas para grafos não orientados
    if data.directed:
        return "A verificação de conjunto dominante atualmente só é aplicável a grafos não orientados."

    G = cria_grafo(orientado=False)  # Certifique-se de criar o grafo como não orientado
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    
    # Conjunto de vértices dominados pelo conjunto fornecido
    dominados = set(conjunto)
    
    # Adiciona os vizinhos de cada vértice no conjunto
    for v in conjunto:
        dominados.update(G.neighbors(v))
    
    # Verifica se todos os vértices do grafo foram dominados
    dominante = len(dominados) == len(G)
    
    return f"O conjunto {conjunto} {'é' if dominante else 'não é'} dominante."



@app.post("/grafo/verificar_planaridade", summary="Verifica se o grafo é planar",
          response_description="Se o grafo é planar")
async def verificar_planaridade(data: EdgeList):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    planar, _ = nx.check_planarity(G)
    return f"O grafo {'é' if planar else 'não é'} planar."


@app.post("/grafo/caminho_mais_curto", summary="Encontra o caminho mais curto entre dois vértices",
          response_description="Caminho mais curto")
async def caminho_mais_curto(data: EdgeList, source: str, target: str):
    print(data)
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    try:
        caminho = nx.shortest_path(G, source, target, method='bellman-ford', weight='weight' if any('weight' in d for _, _, d in G.edges(data=True)) else None)
        return f"O caminho mais curto entre {source} e {target} é: {caminho}."
    except nx.NetworkXNoPath:
        return f"Não existe caminho entre {source} e {target}."


@app.post("/grafo/verificar_arestas_do_no", summary="Verifica se um nó tem arestas e lista as arestas",
          response_description="Arestas conectadas ao nó")
async def verificar_arestas_do_no(data: EdgeList, no: str):
    G = cria_grafo(orientado=data.directed)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    arestas = list(G.edges(no))
    return f"O nó {no} {'tem arestas conectadas: ' + str(arestas) if arestas else 'não tem arestas conectadas.'}"
    

@app.post("/grafo/algoritmo_hungaro_tabela", summary="Implementa o Algoritmo Húngaro em um grafo bipartido completo ponderado",
          response_description="Emparelhamento mínimo de custo")
async def algoritmo_hungaro(input_data: GraphInput):
    try:
        resultado = hungarian_algorithm(input_data.graph)
        return f"{resultado}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grafo/verificar_conexo", summary="Verifica se um grafo não orientado é conexo", response_description="Se o grafo é conexo")
async def verificar_conexo(data: EdgeList):
    if data.directed:
        return "Não é possível fazer a verificação, pois o grafo é orientado."
    
    G = cria_grafo(orientado=False)
    adicionar_arestas(G, data.edges)
    adicionar_nos(G, data.nodes)
    conexo = nx.is_connected(G)
    
    return f"O grafo {'é' if conexo else 'não é'} conexo."


# Monta o diretório estático para servir o arquivo index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Monta o diretório estático para servir os arquivos HTML e JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Exemplo de inicialização do servidor FastAPI
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True, root_path="/p/ezgraph")