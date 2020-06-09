import networkx as nx


class CoreGraph(object):
    """Core class to create a Directed graph using the networkx library.

    The graph created with this class will contain nodes with the following attributes:

        - doc_id : document id that the word belongs to.  
        - pri : position reference id. this is a list of tuples "(sid, pid)" where "sid" is the sentence id and "pid" is the position id of the word in the sentence.  
        - weight : the weight of the edge between two exisiting nodes. It consists of the number of sentence that contains the same sequence of the two words
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def update_doc_id(self, current_node_label, doc_id):
        """Update the documents ids list

        Args:
            current_node_label (str): the label node in the graph.
            doc_id (int): the index of the document treated.

        Returns:
            None.

        """
        self.G._node[current_node_label]["doc_id"].append(doc_id)

    def update_pri(self, current_node_label, sent_id, word_pid):
        """Update the position reference information (sid,pid) list

        Args:
            current_node_label (str): the label node in the graph.
            sent_id (int): the index of the sentence in list of sentences.
            word_pid (int): the index position of the word in the sentence used.

        Returns:
            None.

        """
        self.G._node[current_node_label]["pri"].append((sent_id, word_pid))

    def update_edge_weight(self, node_from, node_to, weight):
        """
        Update the weight of the edge between two exisiting nodes

        Args:
            node_from (str): the label node to start from.
            node_to (str): the label node to go to..
            weight (int): the new weight to use for the graph edge.

        Returns:
            None.

        """
        self.G[node_from][node_to]["weight"] = weight

    def draw_network(self):

        nx.draw_networkx(self.G,
                         with_labels=True,
                         font_size=8,
                         node_size=1500,
                         alpha=0.3,
                         arrows=True)
