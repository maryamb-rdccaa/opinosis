import re
import numpy as np

# from .. import PROPERTIES_LIST
from .utils.node import Node
from .utils.candidate import Candidate
from .core import CoreOpinosis


# PARAMETERS = PROPERTIES_LIST["parameters"]


class BasicSummarizerEnglish(CoreOpinosis):
    def __init__(self,
                 directed_graph,
                 config_max_summaries=20.0,
                 config_vsn_threshold=30,
                 config_min_redundancy=1,
                 config_permissable_gap=3,
                 config_scoring_function=3,
                 config_attachment_after=2,
                 config_normalize_overallgain=True,
                 config_turn_on_collapse=True,
                 config_turn_on_dup_elim=True,
                 #  p_topic_threshold=0.1,
                 #  p_sentence_threshold=0.05,
                 p_max_sent_length=18,
                 p_min_sent_length=2,
                 #  p_min_topic_overlap=1,
                 #  config_use_pos_gain=False,
                 config_duplicate_threshold=0.35,
                 config_duplicate_collapse_threshold=0.6):
        """Custom vanilla version of the Opinosis Summarizer for english text.

        Args:
            directed_graph (graph.core.CoreGraph): a networx Directed graph with nodes and attributes followinf the Opinosis conventions.
            config_max_summaries (float, optional): Maximum number of sentences to generate in the summary. Defaults to 20.0.
            config_vsn_threshold (int, optional): Valid start node threshold. It's compared with the average poistion of the word in the sentences. Defaults to 30.
            config_min_redundancy (int, optional):  The minimum number of overlapping sentences covered by this path to be considered in the summary. Defaults to 1.
            config_permissable_gap (int, optional): This parameter controls the maximum allowed gaps in discovering these redundancies. Defaults to 3.
            config_scoring_function (int, optional): The scoring method for computing the path score.
                                                     if 1 then compute GAIN REDUNDANCY ONLY, 2 for GAIN WEIGHTED REDUNDANCY BY PATH LENGTH
                                                     and 3 for GAIN WEIGHTED REDUNDANCY BY LOG PATH LENGTH. Defaults to 3.
            config_attachment_after (int, optional): Minimum number of words in the Anchor before collapsing extensions. Defaults to 2.
            config_normalize_overallgain (bool, optional): If True, divide the path score with the path length. Defaults to True.
            config_turn_on_collapse (bool, optional): If True, activate the collapsing mechanism. Defaults to True.
            config_turn_on_dup_elim (bool, optional): Whether to eliminate duplicate candidates when using collapsing mechanism (Please see `compute_candidate_sim_score` and `remove_duplicates` methods). Defaults to True.
            p_max_sent_length (int, optional): Maximum length of each sentence in the generated summary. Defaults to 18.
            p_min_sent_length (int, optional): Minimum length of each sentence in the generated summary. Defaults to 2.
            config_duplicate_threshold (float, optional): If similarity score between two candidates is greather than this threshold then keep only the best candidate. Defaults to 0.35.
            config_duplicate_collapse_threshold (float, optional): If similarity score between two INTERMEDIATE candidates is greather than this threshold then keep only the best candidate. Defaults to 0.6.
        """
        super(BasicSummarizerEnglish, self).__init__()
        self.G = directed_graph

    def is_valid_start_node(self, node, vsn_threshold=5):
        """Checks if a node is a Valid Start Node.

        A node Vq is a valid start node if it is a natural starting point
        of a sentence.
        We use the positional information of a node to determine if it is a VSN. 
        Specifically, we check if Average(PIDvq) <= vsn_threshold,
        where vsn_threshold is a parameter to be empirically set.
        With this, we only qualify nodes that tend to occur early on in a sentence.  

        Args:
            node (summarizer.node.Node): a Node object (please see the Node class in the summarizer.node module).
            vsn_threshold (int, optional): Valid start node threshold. It's compared with the average poistion of the word in the sentences. Defaults to 5.

        Returns:
            bool: True if the node is a valid start.
        """

        node_label = node.get_node_label().lower()
        avg_node_pid = node.get_average_position()

        match1, match2 = False, False
        if re.match("^it:*:prp", node_label, re.I):
            match1 = True
        if re.match("^(its:|the:|when:|a:|an:|this:|the:|they:|it:|i:|we:|our:).*", node_label, re.I):
            match2 = True

        expression = node_label.endswith(":jj") | \
            node_label.endswith(":jjr") | \
            node_label.endswith(":jjs") | \
            node_label.endswith(":afx") | \
            node_label.endswith(":rb") | \
            node_label.endswith(":rbr") | \
            node_label.endswith(":rbs") | \
            node_label.endswith(":prp") | \
            node_label.endswith(":vbg") | \
            node_label.endswith(":vbd") | \
            node_label.endswith(":vbn") | \
            node_label.endswith(":vbp") | \
            node_label.endswith(":vbz") | \
            node_label.endswith(":hvs") | \
            node_label.endswith(":bes") | \
            node_label.endswith(":md") | \
            node_label.endswith(":vb") | \
            node_label.endswith(":nn") | \
            node_label.endswith(":nnp") | \
            node_label.endswith(":nnps") | \
            node_label.endswith(":nns") | \
            node_label.endswith(":dt") | \
            node_label.endswith(":pdt") | \
            node_label.endswith(":prp$") | \
            node_label.endswith(":wdt") | \
            node_label.endswith(":wp$") | \
            node_label.startswith("it:") | \
            node_label.startswith("if:") | \
            node_label.startswith("for:") | \
            match1 | \
            match2

        if avg_node_pid <= vsn_threshold:
            if expression:
                return True
        return False

    def is_end_token(self, node):
        """Checks if a node is an End Node.

        Args:
            node (summarizer.node.Node): a Node object (please see the Node class in the summarizer.node module).

        Returns:
            bool: True if the node is a end of the sentence.
        """
        # TODO Add other coordinating conjunction
        VEN_LIST = list('!,.:;?') + ['but', 'yet']
        node_label = node.get_node_label()
        node_text, node_pos_tag, node_tag = node_label.split('::')

        if node_text in VEN_LIST:
            return True
        return False

    def is_valid_end_node(self, node):
        """Checks if a node is a Valid End Node.

        Either an end node or a node that does not have any neighbors.

        Args:
            node (summarizer.node.Node): a Node object (please see the Node class in the summarizer.node module).

        Returns:
            bool: True if the node is a valid end node.
        """
        if self.is_end_token(node):
            return True

        node_label = node.get_node_label()
        if self.G.out_degree(node_label) <= 0:
            return True

        return False

    def is_valid_candidate(self, path_label):
        """Checks if the sentence is a valid one given the defined noun phrases patterns.

        Args:
            path_label (str): the path sentence (ex: This::DET::DT is::VERB:VBZ a::DET::DT sentence::NOUN:NN)

        Returns:
            bool: True if the path is valid candidate for the summary.
        """
        is_good = False

        # (str.matches(".*(/jj)*.*(/nn)+.*(/vb)+.*(/jj)+.*"))
        if re.match(".*(::ADJ::\w+\s)*.*(::NOUN::\w+\s|::PROPN::\w+\s)+.*(::VERB::\w+\s|::AUX::\w+\s)+.*(::ADJ::\w+\s?|::ADV::\w+\s?)+.*", path_label, re.I):
            is_good = True
        else:
            condition1 = re.match(
                ".*(::ADV::\w+\s)*.*(::ADJ::\w+\s)+.*(::NOUN::\w+\s?)+.*", path_label, re.I)
            condition2 = re.match(".*(::DET::\w+\s?).*", path_label, re.I)
            if condition1 and not condition2:
                is_good = True

            # (str.matches(".*(/prp|/dt)+.*(/vb)+.*(/rb|/jj)+.*(/nn)+.*"))
            elif re.match(".*(::PRON::\w+\s|::DET::\w+\s)+.*(::VERB::\w+\s|::AUX::\w+\s)+.*(::ADV::\w+\s|::ADJ::\w+\s)+.*(::NOUN::\w+\s?)+.*", path_label, re.I):
                is_good = True

                #  (str.matches(".*(/jj)+.*(/to)+.*(/vb).*"))
            elif re.match(".*(::ADJ::\w+\s)+.*(::\w+::TO\s?)+.*(::VERB::\w+\s?|::AUX::\w+\s?).*", path_label, re.I):
                is_good = True

                # (str.matches(".*(/rb)+.*(/in)+.*(/nn)+.*"))
            elif re.match(".*(::ADV::.*\s)+.*(::.*::IN\s)+.*(::NOUN::.*\s?)+.*", path_label, re.I):
                is_good = True

            # String last = str.substring(str.lastIndexOf(' '), str.length());
            # if (last.matches(".*(/to|/vbz|/in|/cc|wdt|/prp|/dt|/,)")) {
            #   isGood = false;
            # }

            else:
                tokens = re.split("\s+", path_label)
                last = tokens[-1]
                if re.match("\w+::\w+::(to|vbz|in|cc|wdt|prp|dt|nfp)", last, re.I):
                    is_good = False
        return is_good

    def compute_adjusted_score(self, score, level):
        overall_gain = score
        if self.PARAMETERS["CONFIG_NORMALIZE_OVERALLGAIN"]:
            overall_gain /= level
        return overall_gain

    def compute_score(self, current_path_score, current_overlap_list, path_length):
        overlap_size = len(current_overlap_list)

        if (self.PARAMETERS["CONFIG_SCORING_FUNCTION"] == self.PARAMETERS["GAIN_REDUNDANCY_ONLY"]):
            score = current_path_score + overlap_size

        elif (self.PARAMETERS["CONFIG_SCORING_FUNCTION"] == self.PARAMETERS["GAIN_WEIGHTED_REDUNDANCY_BY_LEVEL"]):
            score = current_path_score + (overlap_size*path_length)

        elif (self.PARAMETERS["CONFIG_SCORING_FUNCTION"] == self.PARAMETERS["GAIN_WEIGHTED_REDUNDANCY_BY_LOG_LEVEL"]):
            if path_length > 1:
                score = current_path_score + (overlap_size*np.log(path_length))
            else:
                score = current_path_score + overlap_size
        return score

    def compute_candidate_sim_score(self, candidate1, candidate2):
        node_list_1 = [node.get_node_label() for node in candidate1.node_list]
        node_list_2 = [node.get_node_label() for node in candidate2.node_list]

        intersect_size = len(set(node_list_1) & set(node_list_2))
        union_size = len(set(node_list_1) | set(node_list_2))
        try:
            sim_score = intersect_size / union_size
        except:
            sim_score = 0
        return sim_score

    def get_node_list(self, sentence):
        node_list = []
        tokens = re.split("\s+", sentence)

        for token in tokens:
            if token == "xx":
                continue
            if re.match("\w+::ADV::\w+|\w+::ADJ::\w+|\w+::NOUN::\w+|\w+::PROPN::\w+|\w+::VERB::\w+|\w+::AUX::\w+", token, re.I):
                node = Node(token, self.G[token])
                if node is not None:
                    node_list.append(node)

        return node_list
