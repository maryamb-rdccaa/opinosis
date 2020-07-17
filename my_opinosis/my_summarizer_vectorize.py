#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:42:57 2020

@author: maryam
"""
#%%
import re
import numpy as np
import copy
import warnings
from collections import defaultdict
import platform
import inspect
from tqdm import tqdm
import pandas as pd

#%%

class Node(object):
    def __init__(self, node_label, node_attributes):
        self.node_label = node_label
        self.node_attributes = node_attributes
    def get_node_label(self):
        return self.node_label
    def get_node_doc_id(self):
        return self.node_attributes["doc_id"]
    def get_node_pri(self):
        return self.node_attributes["pri"]
    def get_average_position(self):
        pri = self.get_node_pri()
        node_pids = [sid_pid_tuple[1] + 1 for sid_pid_tuple in pri]
        avg_node_pid = np.mean(node_pids)
        return avg_node_pid

class Candidate(object):
    def __init__(self, overall_gain=0.0, sentence="", sentence_list=[], level=0, raw_score=0.0, local_gain=0.0, overlap=0.0,  node_list=[], discard=False):
        self.discard = discard
        self.overall_gain = overall_gain
        self.sentence = sentence
        self.sentence_list = sentence_list
        self.level = level
        self.raw_score = raw_score
        self.local_gain = local_gain
        self.overlap = overlap
        self.node_list = node_list
        
    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("the candidate should always "
                                   "specify its parameters in the signature"
                                   " of its __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        params_names = sorted([p.name for p in parameters])
        return params_names

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('Next version, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute.'
                              'For now, it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

#%%
class BasicSummarizerEnglish(object):
    def __init__(self,
                 directed_graph,params,Patterns):
        self.Patterns=Patterns
        self.params=params
        self.config_max_summaries = params['config_max_summaries']
        self.config_vsn_threshold = params['config_vsn_threshold']
        self.config_min_redundancy = params['config_min_redundancy']
        self.config_permissable_gap = params['config_permissable_gap']
        self.config_scoring_function = params['config_scoring_function']
        self.config_attachment_after = params['config_attachment_after']
        self.config_normalize_overallgain = params['config_normalize_overallgain']
        self.config_turn_on_collapse = params['config_turn_on_collapse']
        self.config_turn_on_dup_elim = params['config_turn_on_dup_elim']
        self.p_max_sent_length = params['p_max_sent_length']
        self.p_min_sent_length = params['p_min_sent_length']
        self.config_duplicate_threshold = params['config_duplicate_threshold']
        self.config_duplicate_collapse_threshold = params['config_duplicate_collapse_threshold']

        super(BasicSummarizerEnglish, self).__init__()
        self.G = directed_graph
        self.mAnchor = ""
        self.before_attach_gain = 0.0
        self.m_anchor_path_score = 0.0
        self.m_anchor_path_length = 0

        self.short_listed = []
        self.short_listed_dict = dict()
        self.cc_dict = dict()
        self.debug = False

        self.PARAMETERS = {
            "CONFIG_MAX_SUMMARIES": self.config_max_summaries,
            "CONFIG_VSN_THRESHOLD": self.config_vsn_threshold,
            "CONFIG_MIN_REDUNDANCY": self.config_min_redundancy,
            "CONFIG_PERMISSABLE_GAP": self.config_permissable_gap,
            "CONFIG_SCORING_FUNCTION": self.config_scoring_function,
            "CONFIG_ATTACHMENT_AFTER": self.config_attachment_after,
            "CONFIG_NORMALIZE_OVERALLGAIN": self.config_normalize_overallgain,
            "CONFIG_TURN_ON_COLLAPSE": self.config_turn_on_collapse,
            "CONFIG_TURN_ON_DUP_ELIM": self.config_turn_on_dup_elim,
            "P_MAX_SENT_LENGTH": self.p_max_sent_length,
            "P_MIN_SENT_LENGTH": self.p_min_sent_length,
            "CONFIG_DUPLICATE_THRESHOLD": self.config_duplicate_threshold,
            "CONFIG_DUPLICATE_COLLAPSE_THRESHOLD": self.config_duplicate_collapse_threshold,
            'GAIN_REDUNDANCY_ONLY': 1,
            'GAIN_WEIGHTED_REDUNDANCY_BY_LEVEL': 2,
            'GAIN_WEIGHTED_REDUNDANCY_BY_LOG_LEVEL': 3
        }

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("the object should always "
                                   "specify its parameters in the signature"
                                   " of its __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        params_names = sorted([p.name for p in parameters])
        return params_names

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('Next version, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute.'
                              'For now, it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def get_final_sentences(self):
        def candidate_sort_func(c):
            return c.overall_gain

        output = []
        if len(self.short_listed_dict) <= 0:
            return output
        temp = self.remove_duplicates(self.short_listed_dict, False)
        temp_sorted = sorted(temp, key=candidate_sort_func, reverse=True)
#        temp_sorted = sorted(temp, key=overall_gain, reverse=True)
        if len(temp_sorted) > self.PARAMETERS["CONFIG_MAX_SUMMARIES"]:
            output = temp_sorted[:int(self.PARAMETERS["CONFIG_MAX_SUMMARIES"])] #, candidate_sort_func
        else:
            output = temp_sorted#, candidate_sort_func
        return output

    def start(self):
        """Main function of the object.

        The entry point of the Opinisis summarizer object. This method launches the whole summary process.

        Returns:
            final_sentences_results (list) : List of sentences in the generated summary
        """
        Input=pd.DataFrame([[key, value] for key, value in tqdm(self.G.nodes.data(), desc="Nodes : ")])
#        Input=[[key, value] for key, value in tqdm(self.G.nodes.data(), desc="Nodes : ")]
#        primary_output=pd.DataFrame(index=Input.iloc[:,0],columns=['value'])
#        primary_output=[]
        total_output=dict()
        def primary(v1,v2): #Input[i][0],Input[i][1]
            self.short_listed_dict = dict()
            key, value = v1 , v2
            node = Node(key, value)
            path_score = 0.0
            path_length = 1
            is_collapsible_candidate = False
            overlap_same = False
            if self.is_valid_start_node(node, vsn_threshold=self.PARAMETERS["CONFIG_VSN_THRESHOLD"]):
                self.traverse(node,
                              node.get_node_pri(),
                              node.get_node_label(),
                              path_score,
                              path_length,
                              is_collapsible_candidate,
                              overlap_same
                              )
            v3=self.short_listed_dict
            return v3
        primary_vect=np.vectorize(primary)
        Primary_output=primary_vect(Input.iloc[:,0],Input.iloc[:,1])
#        return Primary_output
#        Primary_output_pd=pd.DataFrame(Primary_output)
        
        for i in range(len(Primary_output)):
            total_output.update(Primary_output[i])  
            
        self.short_listed_dict=total_output
        
        the_sentences_info = self.get_final_sentences()
        #the_sentences_info=list(primary(Input).iloc[:,0])
        def format_output_sentences(sent):
            words = sent.split()
            tokens = [tuple(word.split("::")) for word in words]
            clean_tokens = []
            for word in words:
                if len(word.split("::")) <= 1:
                    clean_tokens.append(word)
                else:
                    word_text, pos_tag, word_tag = tuple(word.split("::"))
                    clean_tokens.append(word_text)
            output_sent = ' '.join(clean_tokens)
            return output_sent

        final_sentences_results = [format_output_sentences(
            s.sentence.replace("xx ", "")) for s in the_sentences_info]

        return final_sentences_results


    def should_countinue_traverse(self, node, overlap_so_far, path_score,
                                  path_length, min_redundancy_threshold,
                                  path_length_threshold):
        if path_length >= path_length_threshold:
            return False
        if path_score == -np.inf:
            return False
        if (len(overlap_so_far) < min_redundancy_threshold and not self.is_end_token(node)):
            return False
        return True

    def process_valid_end_node(self, node, overlap_list, path_label,
                               path_score, path_length,
                               is_collapsible_candidate):
        the_candidate_label = path_label
        the_path_length = path_length
        the_path_score = path_score
        if self.is_end_token(node):
            # remove the last node_label
            try:
                blank =the_candidate_label.rindex(" ")
                the_candidate_label = the_candidate_label[:blank]
            except:
                pass
            the_path_length = max(1, path_length - 1)        
        the_adjusted_score = self.compute_adjusted_score(
            the_path_score, the_path_length)
        if self.is_valid_candidate(self.mAnchor + " " + the_candidate_label):
            if not is_collapsible_candidate:
                candidate = Candidate()
                candidate_params = {'level': the_path_length,'overall_gain': the_adjusted_score,
                                    'sentence': the_candidate_label,'sentence_list': overlap_list}
                candidate.set_params(**candidate_params)
                self.short_listed_dict[the_candidate_label] = candidate
            else:
                candidate = self.cc_dict.get(the_candidate_label, None)
                candidate_path_length = the_path_length - self.m_anchor_path_length
                candidate_path_score = the_path_score - self.m_anchor_path_score
                if candidate is not None:
                    candidate.overall_gain = np.max(
                        [candidate.overall_gain, the_adjusted_score])
                else:
                    candidate = Candidate()
                    candidate_params = {'level': candidate_path_length,'local_gain': 0.0 - self.before_attach_gain,
                                        'overall_gain': the_adjusted_score,'raw_score': candidate_path_score,
                                        'sentence': the_candidate_label,'sentence_list': overlap_list}
                    candidate.set_params(**candidate_params)
                    self.cc_dict[the_candidate_label] = candidate
                return True
        return False

    def get_node_overlap(self, pri_left, pri_right, gap_threshold):#=2):
        pri_output = []
        pointer = 0
        for pri in pri_left:
            if pointer > len(pri_right):
                break
            lag_sid, lag_pid = pri
            for sid, pid in pri_right[pointer:]:
                if sid == lag_sid:
                    if pid-lag_pid > 0 and np.abs(pid-lag_pid) <= gap_threshold:
                        pri_output.append((sid, pid))
                        pointer += 1
                        break
                elif sid > lag_sid:
                    break
                else:
                    pass
        return pri_output

    def do_collapse(self, node, overlap_list, path_label, current_path_score, level, is_collapsible_path, current_overlap_list, path_score):
        self.mAnchor = path_label
        self.m_anchor_path_score = current_path_score
        self.m_anchor_path_length = level
        node_neighbors = list(self.G.neighbors(node.get_node_label()))
        if len(node_neighbors) > 1:
            for neighbor in node_neighbors:
                neighbor_node = Node(neighbor, self.G.nodes.data()[neighbor])
                temp_overlap_list = self.get_node_overlap(pri_left=overlap_list,
                                                          pri_right=neighbor_node.get_node_pri(),
                                                          gap_threshold=self.PARAMETERS["CONFIG_PERMISSABLE_GAP"])
                new_level = level + 1
                new_path_score = self.compute_score(
                    path_score, temp_overlap_list, new_level)
                if len(temp_overlap_list) >= self.PARAMETERS["CONFIG_MIN_REDUNDANCY"]:
                    self.traverse(neighbor_node, temp_overlap_list, "xx " +
                                  neighbor_node.get_node_label(), new_path_score, new_level, True, False)
        is_collapsible_path = False
        success = self.process_found()
        return success

    def remove(self, current_sentence, best):
        if best.overall_gain < current_sentence.overall_gain and best.level <= current_sentence.level:
            best.discard = True
            best = current_sentence
        else:
            current_sentence.discard = True
        return best

    def remove_duplicates(self, candidates_dict, is_intermediate):
        final_sentences = []
        if self.PARAMETERS["CONFIG_TURN_ON_DUP_ELIM"]:
            for candidate_text, candidate in candidates_dict.items():
                candidate.discard = False
                node_list = self.get_node_list(candidate.sentence)
                candidate.node_list = node_list
            for candidate_text, candidate in candidates_dict.items():
                if not candidate.discard:
                    best = candidate
                    for another_candidate_text, another_candidate in candidates_dict.items():
                        if not another_candidate.discard and another_candidate.sentence != candidate.sentence:
                            overlap = self.compute_candidate_sim_score(
                                another_candidate, best)
                            if is_intermediate:
                                if overlap > self.PARAMETERS["CONFIG_DUPLICATE_COLLAPSE_THRESHOLD"]:
                                    best = self.remove(another_candidate, best)
                            else:
                                if overlap > self.PARAMETERS["CONFIG_DUPLICATE_THRESHOLD"]:
                                    best = self.remove(another_candidate, best)
                    final_sentences.append(best)
                    best.discard = True
        else:
            final_sentences = [
                candidate for candidate_text, candidate in candidates_dict.items()]
        return final_sentences

    def process_found(self,):
        success = False
        temp = self.cc_dict
        collapsed = self.remove_duplicates(temp, True)
        i = 0
        if len(collapsed) > 1:
            overall_gains = 0.0
            all_scores = self.m_anchor_path_score
            all_gains = self.before_attach_gain
            all_levels = self.m_anchor_path_length
            buffer = self.mAnchor
            senteces_list = []
            for candidate in collapsed:
                overall_gains += candidate.overall_gain
                all_gains += candidate.local_gain
                all_scores += candidate.raw_score
                all_levels += candidate.level
                senteces_list.append(candidate.sentence_list)
                if (i > 0 and i == (len(collapsed) - 1)):
                    buffer += " et "
                elif i > 0:
                    buffer += " , "
                else:
                    buffer += " , "
                buffer += candidate.sentence
                i += 1
            if len(self.cc_dict) > 1:
                overall_gain = overall_gains / len(self.cc_dict)
                self.short_listed_dict[buffer] = Candidate(
                    overall_gain, buffer, senteces_list, all_levels)
                success = True
        self.mAnchor = ""
        self.before_attach_gain = 0.0
        self.m_anchor_path_score = 0.0
        self.m_anchor_path_length = 0
        self.cc_dict = {}

        return success

    def process_next(self, node, overlap_list, path_label, current_path_score,
                     path_length, is_collapsible_path):
        node_neighbors = list(self.G.neighbors(node.get_node_label()))
        if len(node_neighbors):
            do_more = True
            for neighbor in node_neighbors:
                if not do_more:
                    break
                neighbor_node = Node(neighbor, self.G.nodes.data()[neighbor])
                current_overlap_list = self.get_node_overlap(pri_left=overlap_list,
                                                             pri_right=neighbor_node.get_node_pri(),
                                                             gap_threshold=self.PARAMETERS["CONFIG_PERMISSABLE_GAP"])
                if len(current_overlap_list):
                    new_path_length = path_length + 1
                    new_path_score = self.compute_score(
                        current_path_score, current_overlap_list, new_path_length)
                    node_label = node.get_node_label()
#                    match = re.match(".*:(vb[a-z]|in)$", node_label, re.I)
                    match1 = re.search("VerbForm", node_label, re.I)
                    match2 = re.search(":ADP:", node_label, re.I)
                    match3 = re.search(":ADV:", node_label, re.I)
                    match=match1 or match2 or match3
                    if (self.PARAMETERS["CONFIG_TURN_ON_COLLAPSE"]
                            and path_length >= self.PARAMETERS["CONFIG_ATTACHMENT_AFTER"]
                            and len(current_overlap_list) <= len(overlap_list)
                            and match
                            and not is_collapsible_path):
                        success = self.do_collapse(node, overlap_list, path_label,
                                                   current_path_score, path_length,
                                                   is_collapsible_path,
                                                   current_overlap_list, new_path_score)
                        do_more = False
                        if not success:
                            label = path_label + " " + neighbor_node.get_node_label()
                            do_more = self.traverse(neighbor_node,
                                                    current_overlap_list,
                                                    label,
                                                    new_path_score,
                                                    new_path_length,
                                                    is_collapsible_path,
                                                    False)
                        continue
                    label_temp = path_label + " " + neighbor_node.get_node_label()
                    do_more = self.traverse(neighbor_node, current_overlap_list, label_temp,
                                            new_path_score, new_path_length, is_collapsible_path, False)

    def traverse(self, node, overlap_list, path_label, path_score, path_length,
                 is_collapsible_candidate, overlap_same):
        if not self.should_countinue_traverse(node, overlap_list, path_score,path_length,
                                              min_redundancy_threshold=self.PARAMETERS["CONFIG_MIN_REDUNDANCY"],
                                              path_length_threshold=self.PARAMETERS["P_MAX_SENT_LENGTH"]):
            return True
        if self.is_valid_end_node(node):
            if self.process_valid_end_node(node, overlap_list, path_label,
                                           path_score, path_length,
                                           is_collapsible_candidate):
                return True
        self.process_next(node, overlap_list, path_label, path_score,
                          path_length, is_collapsible_candidate)
        return True

    def is_valid_start_node(self, node, vsn_threshold=5):  #bp changement
        match1=False
        match2=False
        node_label = node.get_node_label().lower()
        avg_node_pid = node.get_average_position()
        if avg_node_pid <= vsn_threshold:
            node_text, node_pos_tag, node_tag = node_label.split('::')
            if node_pos_tag.lower() in ['sconj','det','adv','pron','num']:#,'adp'
                match1=True
            if (node_pos_tag.lower()=='verb')&(node_tag=='VERB__VerbForm=Inf'):
                match2=True
        return match1 | match2

    def is_end_token(self, node): #les changement
        # TODO Add other coordinating conjunction
#        VEN_LIST = list('!,.:;?') + ['but', 'yet']
        VEN_LIST = list(',:!.;?') +['mais' , 'et' ]
        node_label = node.get_node_label()
        node_text, node_pos_tag, node_tag = node_label.split('::')
        if node_text in VEN_LIST:
            return True
#        if node_pos_tag=='CCONJ':
#            return True
        return False

    def is_valid_end_node(self, node):
        if self.is_end_token(node):
            return True
        node_label = node.get_node_label()
        if self.G.out_degree(node_label) <= 0:
            return True
        return False
    
    def is_valid_candidate(self, path_label):
#        print(path_label)
        is_good = False
        sent=path_label.split(' ')
        list_pos=[sent[i].split('::')[1] for i in range(len(sent)) if len(sent[i].split('::'))>1]
        list_tag=[sent[i].split('::')[2] for i in range(len(sent)) if len(sent[i].split('::'))>1]
        if len(list_pos)>2 :
            for Pattern in self.Patterns:
                for i in range(len(list_pos)):
                    if type(Pattern[0][0])!=list :
                        if list_pos[i] in Pattern[0]:
                            Pattern = Pattern[1:]    
                            if len(Pattern) == 0:
                                is_good = True
                                break 
                    else:
                        if [list_pos[i],list_tag[i][-12:]] in Pattern[0] :
                            Pattern = Pattern[1:]
                            if len(Pattern) == 0:
                                is_good = True
                                break          
                if is_good == True:
                    break
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
# %%
