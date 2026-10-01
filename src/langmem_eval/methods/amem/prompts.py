"""Prompts and schemas extracted from WujiangXu/A-mem, MIT licensed.
Upstream commit: 0c8039f28fdcc08189a23c07a3437d9d2482f9c2.
See third_party/amem/LICENSE and README.md for provenance.
"""

ANALYSIS_PROMPT = ('Generate a structured analysis of the following content by:\n'
 '            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)\n'
 '            2. Extracting core themes and contextual elements\n'
 '            3. Creating relevant categorical tags\n'
 '\n'
 '            Format the response as a JSON object:\n'
 '            {\n'
 '                "keywords": [\n'
 '                    // several specific, distinct keywords that capture key concepts and '
 'terminology\n'
 '                    // Order from most to least important\n'
 "                    // Don't include keywords that are the name of the speaker or time\n"
 "                    // At least three keywords, but don't be too redundant.\n"
 '                ],\n'
 '                "context": \n'
 '                    // one sentence summarizing:\n'
 '                    // - Main topic/domain\n'
 '                    // - Key arguments/points\n'
 '                    // - Intended audience/purpose\n'
 '                ,\n'
 '                "tags": [\n'
 '                    // several broad categories/themes for classification\n'
 '                    // Include domain, format, and type tags\n'
 "                    // At least three tags, but don't be too redundant.\n"
 '                ]\n'
 '            }\n'
 '\n'
 '            Content for analysis:\n'
 '            ')

EVOLUTION_PROMPT = ('\n'
 '                                You are an AI memory evolution agent responsible for managing '
 'and evolving a knowledge base.\n'
 '                                Analyze the the new memory note according to keywords and '
 'context, also with their several nearest neighbors memory.\n'
 '                                Make decisions about its evolution.  \n'
 '\n'
 '                                The new memory context:\n'
 '                                {context}\n'
 '                                content: {content}\n'
 '                                keywords: {keywords}\n'
 '\n'
 '                                The nearest neighbors memories:\n'
 '                                {nearest_neighbors_memories}\n'
 '\n'
 '                                Based on this information, determine:\n'
 '                                1. Should this memory be evolved? Consider its relationships '
 'with other memories.\n'
 '                                2. What specific actions should be taken (strengthen, '
 'update_neighbor)?\n'
 '                                   2.1 If choose to strengthen the connection, which memory '
 'should it be connected to? Can you give the updated tags of this memory?\n'
 '                                   2.2 If choose to update_neighbor, you can update the context '
 'and tags of these memories based on the understanding of these memories. If the context and the '
 'tags are not updated, the new context and tags should be the same as the original ones. Generate '
 'the new context and tags in the sequential order of the input neighbors.\n'
 '                                Tags should be determined by the content of these characteristic '
 'of these memories, which can be used to retrieve them later and categorize them.\n'
 '                                Note that the length of new_tags_neighborhood must equal the '
 'number of input neighbors, and the length of new_context_neighborhood must equal the number of '
 'input neighbors.\n'
 '                                The number of neighbors is {neighbor_number}.\n'
 '                                Return your decision in JSON format with the following '
 'structure:\n'
 '                                {{\n'
 '                                    "should_evolve": True or False,\n'
 '                                    "actions": ["strengthen", "update_neighbor"],\n'
 '                                    "suggested_connections": ["neighbor_memory_ids"],\n'
 '                                    "tags_to_update": ["tag_1",..."tag_n"], \n'
 '                                    "new_context_neighborhood": ["new context",...,"new '
 'context"],\n'
 '                                    "new_tags_neighborhood": '
 '[["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],\n'
 '                                }}\n'
 '                                ')

QUERY_PROMPT = ("Given the following question, generate several keywords, using 'cosmos' as the separator.\n"
 '\n'
 '                Question: {question}\n'
 '\n'
 '                Format your response as a JSON object with a "keywords" field containing the '
 'selected text. \n'
 '\n'
 '                Example response format:\n'
 '                {{"keywords": "keyword1, keyword2, keyword3"}}')

ANALYSIS_SCHEMA = {'type': 'object',
 'properties': {'keywords': {'type': 'array', 'items': {'type': 'string'}},
                'context': {'type': 'string'},
                'tags': {'type': 'array', 'items': {'type': 'string'}}},
 'required': ['keywords', 'context', 'tags'],
 'additionalProperties': False}

EVOLUTION_SCHEMA = {'type': 'object',
 'properties': {'should_evolve': {'type': 'boolean'},
                'actions': {'type': 'array', 'items': {'type': 'string'}},
                'suggested_connections': {'type': 'array', 'items': {'type': 'integer'}},
                'new_context_neighborhood': {'type': 'array', 'items': {'type': 'string'}},
                'tags_to_update': {'type': 'array', 'items': {'type': 'string'}},
                'new_tags_neighborhood': {'type': 'array',
                                          'items': {'type': 'array', 'items': {'type': 'string'}}}},
 'required': ['should_evolve',
              'actions',
              'suggested_connections',
              'tags_to_update',
              'new_context_neighborhood',
              'new_tags_neighborhood'],
 'additionalProperties': False}

QUERY_SCHEMA = {'type': 'object',
 'properties': {'keywords': {'type': 'string'}},
 'required': ['keywords'],
 'additionalProperties': False}
