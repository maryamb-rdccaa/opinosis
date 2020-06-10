from tqdm import tqdm

# from .. import PROPERTIES_LIST
from .core import CoreGraph


# NODE_SEPARATOR = PROPERTIES_LIST["node_labels"]["separator"]


class PosTagGraph(CoreGraph):
    def __init__(self, texts, nlp, node_separator="::"):
        """Create an Opinosis style graph

        Create an Opinosis style graph. The node labels are in the form of "Word::POS::TAG".
        The class uses the Part of Speech Taggin in the node label definition.
        Note that the separator "::" can be changed by setting a value for the separator argument.

        Args:
            CoreGraph (object): opinosis Core graph.
            texts (list): list of sentences to be summarized.
            nlp (spacy.lang): SpaCy language model.
            node_separator (str, optional): Separator to be used in the node label. 
                                       If the seperator is "::" then the node label will look like  "Word::POS::TAG".
                                       Defaults to "::".
        """
        super(PosTagGraph, self).__init__()
        self.texts = texts
        self.nlp = nlp
        self.separator = node_separator

    def can_add_to_node(self, node_label):
        """Checks if an edge can be created from the node with the provided label. 

        For now, a simple implementation is used bases on a string comparaison. In other words,
        if the node is a punctuation '.?!' then it should be an end node.

        Args:
            node_label (str): node label as "word:pos_tag". exemple "This::DET::DT"

        Returns:
            bool: Returns True if the node word is different than any of '.?!' and None.

        """
        if node_label is None:
            return False
        else:
            node_word, node_pos_tag, node_tag = node_label.split(
                self.separator)
            if node_word in '.?!':  # TODO use more characters for punctuation as needed
                return False
            else:
                return True

    def is_not_blank(self, word_text, word_pos_tag):
        """Checks if a the word contains only blanks, newlines, tabs or spaces

        Args:
            word_text (str): a string which represente the word to analyze.
            word_pos_tag (str): SpaCy pos tagging.

        Returns:
            bool: Returns True if the node word is different than blank space (SpaCy POS not in ["_SP", "SPACE"]).

        """
        if not word_text.isspace() and word_pos_tag not in ["_SP", "SPACE"]:
            return True
        else:
            return

    def get_word_text_and_pos_tag(self, word):
        """Extract the word text and the pos tag from SpaCy token.

        Args:
            word (spacy.tokens.token.Token): a token from SpaCy document.

        Returns:
            word_text (str): the text of the word in a lower format.
            word_pos_tag (str): the POS tag for the word.

        """
        word_text, word_pos_tag, word_tag = word.lower_.rstrip(), word.pos_, word.tag_
        if word_tag in [".", ",", "LRB-", "-RRB-", "``", '""', "''", ":", "$", "#"]:
            word_tag = "PUNCT"
        return word_text, word_pos_tag, word_tag

    def get_pos_tag_large_list(self, texts, batch_size=300):
        """Process texts as a stream, and yield Doc objects in order.

        This is usually more efficient than processing texts one-by-one if the number of sentences to be processed is large. 

        Args:
            texts (list): list of sentences to be passed to the SpaCy language model to get the words's POS and TAG.
            batch_size (int, optional): Batch size to use with the SpaCy pipe. Defaults to 300.

        Returns:
            sentences_tokens (list): List of tuples (sent_id, word_pid, word_text, word_pos_tag, word_tag) where 
                                        - sent_id : the sentence id that the word belongs to.
                                        - word_pid : position id of the word in the sentence.
                                        - word_text : the word that was processed.
                                        - word_pos_tag : the Part of Speech Tagging obtained by SpaCy's language model.
                                        - word_tag :  the Taggin obtained by SpaCy's language model.
        """
        sentences_tokens = []
        sent_id = 0
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            for word_pid, word in tqdm(enumerate(doc), "Extract POS Tags"):
                word_text, word_pos_tag, word_tag = self.get_word_text_and_pos_tag(
                    word)
                sentences_tokens.append(
                    (sent_id, word_pid, word_text, word_pos_tag, word_tag))
            sent_id += 1
        return sentences_tokens

    def get_pos_tag_small_list(self, texts):
        """Process texts one-by-one using the SpaCy language model.

        It should be used if the number of sentences to be processed is not too big.

        Args:
            texts (list): list of sentences to be passed to the SpaCy language model to get the words's POS and TAG.

        Returns:
            sentences_tokens (list): List of tuples (sent_id, word_pid, word_text, word_pos_tag, word_tag) where 
                                        - sent_id : the sentence id that the word belongs to.
                                        - word_pid : position id of the word in the sentence.
                                        - word_text : the word that was processed.
                                        - word_pos_tag : the Part of Speech Tagging obtained by SpaCy's language model.
                                        - word_tag :  the Taggin obtained by SpaCy's language model.
        """
        sentences_tokens = []
        for sent_id, sent in tqdm(enumerate(texts), desc="Extract POS Tags"):
            doc = self.nlp(str(sent))
            for word_pid, word in enumerate(doc):
                word_text, word_pos_tag, word_tag = self.get_word_text_and_pos_tag(
                    word)
                sentences_tokens.append(
                    (sent_id, word_pid, word_text, word_pos_tag, word_tag))
        return sentences_tokens

    def create_graph(self):
        """The backbone method to create the Opinosis graph.

        First, it processes the sentences to get the POS and Tag of every word using the SpaCy language model.
        It creates the Opinosis directed graph:  

            - node labels as "Word::POS::TAG" 
            - node attributes as {"weigh": w1, "pri": [(sid1, pid1), (sid2, pid2), ... , (sidn, pidn)], doc_id= [id1, id2, ... , idn]}

        Returns:
            G : networkx directed graph
        """
        previous_node_label = None
        is_pervious_node_new = True

        if len(self.texts) > 5000:
            sentences_tokens = self.get_pos_tag_large_list(
                self.texts, batch_size=300)
        else:
            sentences_tokens = self.get_pos_tag_small_list(self.texts)

        for token in tqdm(sentences_tokens, desc="Create Graph"):
            sent_id, word_pid, word_text, word_pos_tag, word_tag = token
            # node label is not only blanks, spaces, tabs, newlines,...
            if self.is_not_blank(word_text, word_pos_tag):
                current_node_label = '{}{}{}{}{}'.format(
                    word_text, self.separator, word_pos_tag, self.separator, word_tag)
                is_current_node_new = True
                # node already exists in the graph
                if self.G.has_node(current_node_label):
                    is_current_node_new = False
                    self.update_doc_id(current_node_label, 1)
                    self.update_pri(current_node_label, sent_id, word_pid)
                # node does not exists in the graph
                else:
                    is_current_node_new = True
                    self.G.add_node(current_node_label, doc_id=[
                                    1], pri=[(sent_id, word_pid)])

                # create an edge between the two nodes (at least one of them is new)
                if (is_current_node_new or is_pervious_node_new):
                    if (previous_node_label is not None):
                        if (current_node_label != previous_node_label and self.can_add_to_node(previous_node_label)):
                            self.G.add_edge(
                                previous_node_label, current_node_label, weight=1)
                # both nodes (current and previous) already exists
                else:
                    # update the weight of existing edge
                    if self.G.has_edge(previous_node_label, current_node_label):
                        edge_data = self.G.get_edge_data(
                            previous_node_label, current_node_label)
                        new_weight = edge_data["weight"] + 1
                        self.update_edge_weight(
                            previous_node_label, current_node_label, new_weight)
                    # Create the edge between the two nodes since it does not exists
                    else:
                        try:
                            if (current_node_label != previous_node_label and self.can_add_to_node(previous_node_label)):
                                self.G.add_edge(
                                    previous_node_label, current_node_label, weight=1)
                        except:
                            print("Problem Linking '{}' and '{}'".format(
                                previous_node_label, current_node_label))
            # node label is only blanks, spaces, tabs, newlines,...
            else:
                word_pid -= 1

            previous_node_label = current_node_label
            is_pervious_node_new = is_current_node_new

        G = self.G
        return G
