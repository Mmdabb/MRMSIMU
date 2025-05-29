import networkx as nx
import pandas as pd

def compute_scc_id_mapping(links_df):
    """
    Compute strongly connected components and return a node-to-SCC index map.

    Returns:
        dict: {node_id: scc_id}, where 0 is the largest SCC
    """
    G = nx.DiGraph()
    for _, row in links_df.iterrows():
        G.add_edge(row['from_node_id'], row['to_node_id'])

    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    return {node_id: idx for idx, scc in enumerate(sccs) for node_id in scc}

def annotate_nodes_with_scc(nodes_df, node_to_scc):
    """
    Add a 'scc_id' column to nodes DataFrame based on mapping.

    Returns:
        pd.DataFrame: Annotated nodes
    """
    nodes_df = nodes_df.copy()
    nodes_df['scc_id'] = nodes_df['node_id'].map(node_to_scc).fillna(-1).astype(int)
    return nodes_df

def assign_and_export_scc(meso_links, meso_nodes, output_csv_path):
    """
    Assign SCC indices to meso_nodes and export to CSV.

    Returns:
        pd.DataFrame: Annotated meso_nodes
    """
    node_to_scc = compute_scc_id_mapping(meso_links)
    meso_nodes_annotated = annotate_nodes_with_scc(meso_nodes, node_to_scc)
    meso_nodes_annotated.to_csv(output_csv_path, index=False)
    print(f"Exported meso_nodes with SCC IDs to {output_csv_path}")
    return meso_nodes_annotated
