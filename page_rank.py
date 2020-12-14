def page_rank(G, d=0.85, tol=0.01, max_iter=100):
    """ Returns the page rank of all nodes in a network graph 

    :param networkx graph G: Directed graph
    :param float d: Damping factor
    :param flat tol: Tolerance for the algorithm convergence
    :param int max_iter: Max iterations
    """
    # 'Pages' in a network and the their count
    nodes = G.nodes()
    n = G.number_of_nodes()
    
    # Adjacency matrix
    a_matrix = nx.adjacency_matrix(G)
    
    # Out degree per node
    out_degree = a_matrix.sum(axis=1)
    
    # Generate the weighted adjacency matrix
    wa_matrix = np.nan_to_num((a_matrix / out_degree).transpose())
    
    # Initial state vector / page rank
    pr =  1./n * np.ones(n).reshape(n, 1) 

    # Get all dangling pages
    a = np.array([0 if i > 0 else 1 for i in out_degree])
    
    # Calculate matrix
    A = d * (wa_matrix + 1./n * np.outer(np.ones(n).reshape(n,1), a)) + (1-d) * 1./n * np.outer(np.ones(n).reshape(n,1), np.ones(n))

    # Page Rank calculation
    for i in range(max_iter):
        # Create temp page rank
        pr_temp = pr[:]
        
        # Calculate Page Rank
        pr = A @ pr
        
        # Check convergence
        err = np.absolute(pr - pr_temp).sum()
        
        if (err < tol):
            return pr
        
    raise Exception(f'PageRank couln't be calculated after {max_iter} iterations. (err={err} > tol = {tol})')
