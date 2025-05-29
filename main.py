import pandas as pd
from pathlib import Path

# === Import functions from modules ===
from network_io import load_node_link_layers
from scc import assign_and_export_scc
from connector_utils import (
    find_nearest_scc_nodes,
    map_taz_to_micro_nodes,
    prepare_and_generate_connectors,
    merge_network_layers
)

def main():
    # === Set file paths ===
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    meso_node_path = input_dir / "mesonet/node.csv"
    meso_link_path = input_dir / "mesonet/link.csv"
    micro_node_path = input_dir / "micronet/node.csv"
    micro_link_path = input_dir / "micronet/link.csv"
    taz_path = input_dir / "deer-valley-taz-centroids.csv"

    # === Step 1: Load layers ===
    layers = load_node_link_layers(
        meso_node_path, meso_link_path,
        micro_node_path, micro_link_path
    )
    meso_nodes = layers['meso_nodes']
    meso_links = layers['meso_links']
    micro_nodes = layers['micro_nodes']
    micro_links = layers['micro_links']
    taz_df = pd.read_csv(taz_path)

    # === Step 2: Assign SCC IDs ===
    meso_nodes = assign_and_export_scc(meso_links, meso_nodes, output_dir / "meso_nodes_with_scc.csv")

    # === Step 3: Find candidate meso nodes for TAZ connectors ===
    taz_with_candidates = find_nearest_scc_nodes(taz_df, meso_nodes, num_candidates=1)

    # === Step 4: Map TAZ to starting micro nodes ===
    taz_micro_map = map_taz_to_micro_nodes(taz_with_candidates, meso_links, micro_links, micro_nodes)

    # === Step 5: Generate connectors and reindexed nodes ===
    (
        updated_meso_nodes, updated_micro_nodes, taz_nodes,
        meso_connector_links, micro_connector_links,
        node_shift_meso, node_shift_micro,
        link_shift_meso
    ) = prepare_and_generate_connectors(
        taz_with_candidates,
        meso_nodes,
        micro_nodes,
        taz_micro_map
    )

    # === Step 6: Merge all layers ===
    final_links, final_nodes = merge_network_layers(
        meso_links, meso_connector_links,
        micro_links, micro_connector_links,
        updated_meso_nodes, updated_micro_nodes, taz_nodes,
        node_shift_meso, node_shift_micro,
        link_shift_meso
    )

    # === Step 7: Export merged outputs ===
    final_links.to_csv(output_dir / "final_links.csv", index=False)
    final_nodes.to_csv(output_dir / "final_nodes.csv", index=False)
    print("Network merge and connector generation complete.")

if __name__ == "__main__":
    main()
