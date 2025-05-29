import pandas as pd

def load_node_link_layers(meso_node_path, meso_link_path, micro_node_path, micro_link_path):
    """
    Load meso and micro node/link CSV files into DataFrames with basic column validation.

    Returns:
        dict: Dictionary containing four DataFrames:
              'meso_nodes', 'meso_links', 'micro_nodes', 'micro_links'
    """

    def _validate_columns(df, required_cols, name):
        """Ensure the DataFrame contains all required columns."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    meso_nodes = pd.read_csv(meso_node_path)
    meso_links = pd.read_csv(meso_link_path)
    micro_nodes = pd.read_csv(micro_node_path)
    micro_links = pd.read_csv(micro_link_path)

    _validate_columns(meso_nodes, ['node_id', 'x_coord', 'y_coord'], "Meso Node CSV")
    _validate_columns(meso_links, ['link_id', 'from_node_id', 'to_node_id'], "Meso Link CSV")
    _validate_columns(micro_nodes, ['node_id', 'x_coord', 'y_coord', 'meso_link_id', 'lane_no'], "Micro Node CSV")
    _validate_columns(micro_links, ['link_id', 'from_node_id', 'to_node_id', 'meso_link_id', 'lane_no'], "Micro Link CSV")

    return {
        'meso_nodes': meso_nodes,
        'meso_links': meso_links,
        'micro_nodes': micro_nodes,
        'micro_links': micro_links
    }
