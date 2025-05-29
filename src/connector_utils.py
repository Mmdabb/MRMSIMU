import pandas as pd
import numpy as np

def find_nearest_scc_nodes(taz_df, meso_nodes, num_candidates=1):
    """
    For each TAZ, find the closest meso nodes in the largest SCC.

    Returns:
        pd.DataFrame: TAZs with added candidate node columns
    """
    largest_scc_nodes = meso_nodes[meso_nodes['scc_id'] == 0].copy()
    result = []

    for _, taz in taz_df.iterrows():

        # largest_scc_nodes['distance'] = np.linalg.norm(
        #     largest_scc_nodes[['x_coord', 'y_coord']].values - taz[['x_coord', 'y_coord']].values, axis=1
        # )

        largest_scc_nodes['distance'] = np.sqrt(
            (largest_scc_nodes['x_coord'] - taz['x_coord']) ** 2 + (largest_scc_nodes['y_coord'] - taz['y_coord']) ** 2
        )

        nearest = largest_scc_nodes.nsmallest(num_candidates, 'distance')
        row = taz.to_dict()
        for i, node_id in enumerate(nearest['node_id']):
            row[f'candidate_node_id_{i + 1}'] = node_id
        result.append(row)

    return pd.DataFrame(result)

def find_start_micro_nodes(meso_node_id, meso_links, micro_links):
    """
    Get micro start nodes for outgoing meso links from a given meso node.

    Returns:
        list: Unique micro node IDs that are only 'from' nodes
    """
    meso_link_ids = meso_links[meso_links['from_node_id'] == meso_node_id]['link_id']
    micro_subset = micro_links[micro_links['meso_link_id'].isin(meso_link_ids)]
    from_nodes = set(micro_subset['from_node_id'])
    to_nodes = set(micro_subset['to_node_id'])
    return list(from_nodes - to_nodes)

def map_taz_to_micro_nodes(taz_candidates, meso_links, micro_links, micro_nodes, candidate_col='candidate_node_id_1'):
    """
    Map each TAZ to micro nodes that start downstream from the candidate meso node.

    Returns:
        pd.DataFrame: Mapping of TAZ to micro node coordinates
    """
    results = []
    for _, row in taz_candidates.iterrows():
        taz_id = row['TAZ_clean']
        meso_node_id = row[candidate_col]
        start_nodes = find_start_micro_nodes(meso_node_id, meso_links, micro_links)
        coords = micro_nodes[micro_nodes['node_id'].isin(start_nodes)]

        for _, node in coords.iterrows():
            results.append({
                'taz_id': taz_id,
                'micro_node_id': node['node_id'],
                'x_coord': node['x_coord'],
                'y_coord': node['y_coord']
            })

    return pd.DataFrame(results)

def compute_shifts(taz_df, meso_nodes):
    """
    Compute shift values for node and link IDs to avoid overlap.

    Returns:
        tuple: (node_shift_meso, node_shift_micro, link_shift_meso)
    """
    max_taz_id = taz_df['TAZ_clean'].max()
    node_shift_meso = max_taz_id + 1
    node_shift_micro = node_shift_meso + meso_nodes['node_id'].max()
    link_shift_meso = len(taz_df) * 2 + 1
    return node_shift_meso, node_shift_micro, link_shift_meso

def reindex_nodes(meso_nodes, micro_nodes, node_shift_meso, node_shift_micro, link_shift_meso):
    """
    Apply node and link ID shifts to meso and micro nodes.

    Returns:
        tuple: (shifted_meso_nodes, shifted_micro_nodes)
    """
    meso_nodes = meso_nodes.copy()
    micro_nodes = micro_nodes.copy()
    meso_nodes['node_id'] += node_shift_meso
    micro_nodes['node_id'] += node_shift_micro
    micro_nodes['meso_link_id'] += link_shift_meso
    meso_nodes['zone_id'] = ''
    micro_nodes['zone_id'] = ''
    return meso_nodes, micro_nodes

def create_taz_nodes(taz_df, taz_id_col):
    """
    Create a TAZ node DataFrame with geometry and zone ID.

    Returns:
        pd.DataFrame: TAZ nodes
    """
    return pd.DataFrame({
        'node_id': taz_df[taz_id_col],
        'x': taz_df['x_coord'],
        'y': taz_df['y_coord'],
        'zone_id': taz_df[taz_id_col],
        'layer': 'taz'
    })

def generate_meso_connectors(taz_df, meso_nodes, candidate_col, taz_id_col, node_shift_meso):
    """
    Generate bidirectional meso-level connector links from TAZ to meso nodes.

    Returns:
        pd.DataFrame: Meso connector links
    """
    connectors = []
    link_id = 1

    for _, row in taz_df.iterrows():
        taz_id = row[taz_id_col]
        taz_x, taz_y = row['x_coord'], row['y_coord']
        meso_node_id = row[candidate_col] + node_shift_meso
        mx, my = meso_nodes[meso_nodes['node_id'] == meso_node_id].iloc[0][['x_coord', 'y_coord']]

        for from_id, to_id, geom in [
            (taz_id, meso_node_id, f"LINESTRING ({taz_x} {taz_y}, {mx} {my})"),
            (meso_node_id, taz_id, f"LINESTRING ({mx} {my}, {taz_x} {taz_y})")
        ]:
            connectors.append({
                "link_id": link_id,
                "from_node_id": from_id,
                "to_node_id": to_id,
                "dir_flag": 1,
                "length": 100,
                "lanes": 1,
                "free_speed": 120,
                "capacity": 100000,
                "link_type_name": "connector",
                "link_type": 0,
                "layer": "meso",
                "geometry": geom,
                "allowed_uses": "auto",
                "from_biway": 1,
                "is_link": 0
            })
            link_id += 1

    return pd.DataFrame(connectors)

def generate_micro_connectors(taz_df, taz_micro_map, micro_nodes, taz_id_col, node_shift_micro, link_id=1):
    """
    Generate bidirectional micro-level connector links from TAZ to micro nodes.

    Returns:
        pd.DataFrame: Micro connector links
    """
    connectors = []
    used_nodes = set()
    link_id = link_id

    for _, row in taz_micro_map.iterrows():
        taz_id = row['taz_id']
        micro_node_id = row['micro_node_id'] + node_shift_micro
        if micro_node_id in used_nodes:
            continue

        taz_row = taz_df[taz_df[taz_id_col] == taz_id].iloc[0]
        taz_x, taz_y = taz_row['x_coord'], taz_row['y_coord']
        mx, my = row['x_coord'], row['y_coord']

        for from_id, to_id, geom in [
            (taz_id, micro_node_id, f"LINESTRING ({taz_x} {taz_y}, {mx} {my})"),
            (micro_node_id, taz_id, f"LINESTRING ({mx} {my}, {taz_x} {taz_y})")
        ]:
            connectors.append({
                "link_id": link_id,
                "from_node_id": from_id,
                "to_node_id": to_id,
                "dir_flag": 1,
                "length": 100,
                "lanes": 1,
                "free_speed": 120,
                "capacity": 100000,
                "link_type_name": "connector",
                "link_type": 0,
                "layer": "micro",
                "geometry": geom,
                "allowed_uses": "auto",
                "from_biway": 1,
                "is_link": 0
            })
            link_id += 1

        used_nodes.add(micro_node_id)

    return pd.DataFrame(connectors)



def prepare_and_generate_connectors(
    taz_df, meso_nodes, micro_nodes, taz_micro_map,
    candidate_col='candidate_node_id_1', taz_id_col='TAZ_clean'
):
    """
    Master function to reindex nodes, create TAZ nodes, and generate meso/micro connector links.

    Returns:
        tuple:
            updated_meso_nodes, updated_micro_nodes, taz_nodes,
            meso_connectors_df, micro_connectors_df,
            node_shift_meso, node_shift_micro,
            link_shift_meso, link_shift_micro
    """
    node_shift_meso, node_shift_micro, link_shift_meso = compute_shifts(taz_df, meso_nodes)

    updated_meso_nodes, updated_micro_nodes = reindex_nodes(
        meso_nodes, micro_nodes, node_shift_meso, node_shift_micro, link_shift_meso
    )

    taz_nodes = create_taz_nodes(taz_df, taz_id_col)

    meso_connectors_df = generate_meso_connectors(
        taz_df, updated_meso_nodes, candidate_col, taz_id_col, node_shift_meso
    )

    micro_connectors_df = generate_micro_connectors(
        taz_df, taz_micro_map, micro_nodes, taz_id_col, node_shift_micro
    )


    return (
        updated_meso_nodes, updated_micro_nodes, taz_nodes,
        meso_connectors_df, micro_connectors_df,
        node_shift_meso, node_shift_micro,
        link_shift_meso
    )


def merge_network_layers(
    meso_links, meso_connector_links,
    micro_links, micro_connector_links,
    meso_nodes, micro_nodes, taz_nodes,
    node_shift_meso, node_shift_micro,
    link_shift_meso
):
    """
    Merge meso/micro/taz nodes and links into unified DataFrames.

    Returns:
        tuple: (final_links_df, final_nodes_df)
    """
    for df, layer in [(meso_nodes, 'meso'), (micro_nodes, 'micro'), (taz_nodes, 'taz')]:
        df['layer'] = layer

    for df, layer in [(meso_links, 'meso'), (micro_links, 'micro')]:
        df['layer'] = layer

    meso_links['from_node_id'] += node_shift_meso
    meso_links['to_node_id'] += node_shift_meso
    meso_links['link_id'] += link_shift_meso

    micro_links['from_node_id'] += node_shift_micro
    micro_links['to_node_id'] += node_shift_micro
    micro_links['meso_link_id'] += link_shift_meso

    link_shift_connector_micro = meso_links['link_id'].max()
    micro_connector_links['link_id'] += link_shift_connector_micro
    link_shift_micro = micro_connector_links['link_id'].max() + 1
    micro_links['link_id'] += link_shift_micro

    final_nodes = pd.concat([taz_nodes, meso_nodes, micro_nodes], ignore_index=True)
    final_nodes = final_nodes[['node_id', 'x', 'y', 'zone_id', 'layer']].sort_values('node_id').reset_index(drop=True)

    final_links = pd.concat([
        meso_connector_links,
        meso_links,
        micro_connector_links,
        micro_links
    ], ignore_index=True)

    ordered_cols = ['link_id', 'from_node_id', 'to_node_id', 'link_type', 'layer']
    extra_cols = [col for col in final_links.columns if col not in ordered_cols]
    final_links = final_links[ordered_cols + extra_cols].sort_values(by=['from_node_id', 'to_node_id']).reset_index(drop=True)

    return final_links, final_nodes
