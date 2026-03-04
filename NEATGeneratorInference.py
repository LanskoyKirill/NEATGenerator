import json
import numpy as np
import os

class NeatModel:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['vocabulary']
        self.vocab_size = data['vocabSize']
        self.neuron_count = data['neuronCount']
        
        self.bias_index = 0
        self.phase_index = 1
        self.input_start = 2 
        self.output_start = data['outputStart']
        
        self.order = data['order']
        self.neuron_has_ff_incoming = data['neuronHasFFIncoming']  # не используется, но оставлено
        self.rnn_decay = data.get('rnnDecay', 0.95)

        # Константы токенов
        self.PAD = data.get('padToken', 0)
        self.SOS = data.get('sosToken', 1)
        self.EOS = data.get('eosToken', 2)
        self.UNK = data.get('unkToken', 3)

        self.connections = data['connections']

        # Список смежности для Feed-Forward (enabled и не recurrent)
        self.adj_list = [[] for _ in range(self.neuron_count)]
        for conn in self.connections:
            if conn['enabled'] and not conn['recurrent']:
                self.adj_list[conn['from']].append((conn['to'], conn['weight']))

    def activation(self, x):
        return np.tanh(x)

    def step(self, token_id, phase, rnn_state):
        """
        Один шаг обработки.
        token_id: текущий входной токен
        phase: фаза (0 – ввод, 1 – генерация)
        rnn_state: состояние RNN с предыдущего шага (массив длины neuron_count)
        Возвращает:
            neurons: значения нейронов после прямого прохода
            next_rnn_state: состояние RNN для следующего шага
        """
        neurons = np.array(rnn_state, dtype=float)
        
        # Устанавливаем входные значения
        neurons[self.bias_index] = 1.0
        neurons[self.phase_index] = phase
        
        # Сбрасываем старые входы токенов
        input_range_end = self.input_start + self.vocab_size
        neurons[self.input_start : input_range_end] = 0.0
        
        current_input_idx = self.input_start + token_id
        if current_input_idx < self.neuron_count:
            neurons[current_input_idx] = 1.0

        # Прямой проход (Feed-Forward)
        for n in self.order:
            # Активируем все нейроны, кроме входных (bias, phase, входные токены)
            if n >= self.input_start + self.vocab_size:
                neurons[n] = self.activation(neurons[n])
            
            for to_node, weight in self.adj_list[n]:
                neurons[to_node] += neurons[n] * weight

        # Рекуррентные связи для следующего состояния
        next_rnn_state = self.rnn_decay * rnn_state
        for conn in self.connections:
            if conn['enabled'] and conn['recurrent']:
                next_rnn_state[conn['to']] += neurons[conn['from']] * conn['weight']

        return neurons, next_rnn_state

    def predict_next(self, context_tokens):
        """
        Предсказывает следующий токен, прогоняя весь контекст с нуля.
        context_tokens: список токенов (включая seed)
        Возвращает ID предсказанного токена.
        """
        rnn_state = np.zeros(self.neuron_count)
        # Проходим все токены, кроме последнего, с фазой 0
        for token in context_tokens[:-1]:
            _, rnn_state = self.step(token, phase=0.0, rnn_state=rnn_state)
        # Последний токен с фазой 1
        neurons, _ = self.step(context_tokens[-1], phase=1.0, rnn_state=rnn_state)
        
        # Берём выходные нейроны
        out_start = self.output_start
        out_end = out_start + self.vocab_size
        logits = neurons[out_start:out_end]
        if len(logits) == 0:
            return self.UNK
        return int(np.argmax(logits))

    def generate(self, seed_text, max_length=20, debug=False):
        """
        Генерация текста с полным пересчётом контекста на каждом шаге (O(n^2)).
        seed_text: начальная строка.
        max_length: максимальное количество генерируемых токенов.
        debug: печатать отладочную информацию.
        Возвращает сгенерированную строку (без seed).
        """
        seed_tokens = self.tokenize(seed_text)
        if not seed_tokens:
            seed_tokens = [self.SOS]
        
        context = seed_tokens.copy()
        generated = []
        
        for step in range(max_length):
            next_token = self.predict_next(context)
            if debug:
                word = self.vocab[next_token] if next_token < len(self.vocab) else "???"
                print(f"Step {step}: Predicted '{word}' (id:{next_token})")
            
            if next_token == self.EOS:
                break
            
            generated.append(next_token)
            context.append(next_token)  # добавляем в контекст для следующего шага
        
        return self.detokenize(generated)

    def tokenize(self, text):
        words = text.lower().split()
        v_map = {word: i for i, word in enumerate(self.vocab)}
        return [v_map.get(w, self.UNK) for w in words]

    def detokenize(self, tokens):
        words = []
        for t in tokens:
            if t == self.EOS:
                break
            if 0 <= t < len(self.vocab):
                words.append(self.vocab[t])
        return " ".join(words)


# --- Пример запуска ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'best_model.json')

    model = NeatModel(json_path)

    listOfRequests = ["The brown dog", "They observe birds in", "A gentle breeze", "However, understanding patience", "Building connections with others", "Many people walk their pets", "Such interactions improve"]
    #listOfRequests = ["Dog in the"]
    answers = []
    for i in range(len(listOfRequests)):
        input_str = listOfRequests[i]
        result = model.generate(input_str, max_length=10, debug=True)
        answers.append((input_str, result))
    for i in range(len(listOfRequests)):
        print(f"\nInput: {answers[i][0]}")
        print(f"Generated: {answers[i][1]}")