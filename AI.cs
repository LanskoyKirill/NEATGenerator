using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Unity.VisualScripting;

public class AI : MonoBehaviour
{
    public GameObject spawnerOfNN;
    public bool go = true;
    public int cIterator = 0;
    public int addition = 0;

    // NEAT параметры
    public float speed = 0.1f;
    public int recursionAddLink = 0;
    public float howIsGood = 0;
    public float textHowIsGood = 0;
    private float accumulatedFitness = 0f;
    private int processedInBatch = 0;

    private float accumulatedNewFitness = 0f;   // сумма фитнеса на новых фразах
    private int processedNewInBatch = 0;        // количество шагов на новых фразах
    public float newPhraseFitness = 0f;         // среднее за батч на новых фразах (для удобства)
    private int maxIterations = 20;

    // Структура сети
    public List<float> neurones = new List<float>();
    public List<int> inpInnov = new List<int>();
    public List<int> outInnov = new List<int>();
    public List<float> weights = new List<float>();
    public List<bool> actConnect = new List<bool>();
    public List<bool> RNNs = new List<bool>();
    public List<float> RNNneurones = new List<float>();
    public List<int> order = new List<int>();
    public List<Dictionary<int, float>> adjList = new List<Dictionary<int, float>>();
    public Dictionary<int, bool> neuronHasFFIncoming = new Dictionary<int, bool>();
    public List<int> innovations = new List<int>();

    public int testing = 0;

    // Данные
    public string inputPhrase;
    public string targetPhrase;
    public string generatedText;
    public bool ifNew = false;
    
    // Токенизированные данные
    public List<int> inputTokens = new List<int>();
    public List<int> targetTokens = new List<int>();
    public List<int> outputTokens = new List<int>();

    public int prevNumber = 0;
    public int thisNumber = 0;
    private int outConnections = 1;
    private int initalNeurones;
    public int desiredNeurones;

    // Состояние для поэтапной подачи
    public int minContextLength = 3;   // минимальная длина контекста перед оценкой
    public int seedLength = 4;               // задаётся GameManager
    private int seedIndex = 0;                // текущий индекс в seedTokens
    private bool generationComplete = false;  // флаг завершения генерации
    private List<int> currentContext = new List<int>();
    private bool seedProcessed = false;
    private List<int> seedTokens = new List<int>();
    private int expectedTokenId = -1;
    
    // Токенизатор
    [NonSerialized] public Tokenizer tokenizer;
    [NonSerialized] public int vocabSize;
    
    private HashSet<int> availableTokens = new HashSet<int>();
    private List<int> availableInputNeurons = new List<int>();
    private List<int> availableOutputNeurons = new List<int>();
    public void Start()
    {
        go = true;
        generationComplete = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        generatedText = "";
        outputTokens.Clear();
        ResetBatchStats();

        // 0: bias (всегда 1)
        // 1: phase (0 - ввод, 1 - генерация)
        // 2..1+vocabSize: входные токены
        // 2+vocabSize..1+2*vocabSize: выходные токены
        // далее скрытые нейроны
        initalNeurones = 2 + vocabSize + vocabSize; // 2 + 2*vocabSize

        if (neurones.Count < initalNeurones)
        {
            neurones.Clear();
            RNNneurones.Clear();

            neurones.Add(1f);               // bias
            RNNneurones.Add(1f);
            neurones.Add(0f);                // phase
            RNNneurones.Add(0f);

            for (int i = 0; i < vocabSize; i++) // входные токены
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }

            for (int i = 0; i < vocabSize; i++) // выходные токены
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }
        }
        else
        {
            // Если сеть уже существовала, обнуляем все нейроны, но bias и phase установятся позже
            for (int i = 0; i < neurones.Count; i++)
            {
                neurones[i] = 0f;
                RNNneurones[i] = 0f;
            }
            neurones[0] = 1f;
            RNNneurones[0] = 1f;
        }

        outConnections = 2 + vocabSize; // начало выходных нейронов

        // Удаление невалидных связей
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (inpInnov[i] >= neurones.Count || outInnov[i] >= neurones.Count)
            {
                inpInnov.RemoveAt(i);
                outInnov.RemoveAt(i);
                weights.RemoveAt(i);
                actConnect.RemoveAt(i);
                RNNs.RemoveAt(i);
                innovations.RemoveAt(i);
                i--;
            }
        }

        UpdateAvailableNeurons(availableTokens);
        makeOrder();
        Adder();

        Material AIMaterial = Resources.Load(spawnerOfNN.name, typeof(Material)) as Material;
        if (AIMaterial != null)
            gameObject.GetComponent<Renderer>().material = AIMaterial;

        if (order.Count > 3 && order[0] == 1 && order[1] == 0 && order[2] == 0)
            ++testing;

        // Проверка целевой последовательности
        if (targetTokens == null || targetTokens.Count < 3)
        {
            Debug.LogWarning("Target tokens too short, generation disabled");
            generationComplete = true;
            go = false;
        }
    }

    void Update()
    {
        if (!inpInnov.Any())
        {
            Destroy(gameObject);
            return;
        }

        if (go && !generationComplete)
        {
            ProcessStep();
        }
    }

    private void ProcessStep()
    {
        int totalTargetLength = targetTokens.Count;
        if (totalTargetLength < 3)
        {
            generationComplete = true;
            go = false;
            return;
        }

        int firstPredictionIndex = seedLength;
        int lastPredictionIndex = totalTargetLength - 2;

        if (firstPredictionIndex < 1) firstPredictionIndex = 1;
        if (firstPredictionIndex > lastPredictionIndex)
        {
            generationComplete = true;
            go = false;
            return;
        }

        for (int pos = firstPredictionIndex; pos <= lastPredictionIndex; pos++)
        {
            ResetNeurons();

            // Подаём контекст (все правильные токены от 1 до pos)
            for (int ctxIdx = 1; ctxIdx <= pos; ctxIdx++)
            {
                int token = targetTokens[ctxIdx];

                for (int j = 2; j < 2 + vocabSize; j++)
                    neurones[j] = 0f;
                int inputIdx = 2 + token;
                if (inputIdx < neurones.Count)
                    neurones[inputIdx] = 1f;

                neurones[1] = 0f; // phase = 0
                RunNetwork();
                UpdateRNNState();
            }

            // Предсказание следующего токена
            neurones[1] = 1f; // phase = 1
            RunNetwork();

            int predictedToken = GetPredictedToken();
            outputTokens.Add(predictedToken);

            int expectedToken = targetTokens[pos + 1];
            float fitness = 0f;

            if (predictedToken == expectedToken)
            {
                fitness = 1f;
            }
            else
            {
                float[] outputActivations = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                {
                    int idx = outConnections + i;
                    outputActivations[i] = idx < neurones.Count ? neurones[idx] : 0f;
                }
                float temperature = 0.5f;
                float[] probs = Softmax(outputActivations, temperature);
                if (expectedToken >= 0 && expectedToken < vocabSize)
                    fitness = probs[expectedToken];
            }

            if (ifNew)
            {
                //Raise the importance of learning new words
                accumulatedNewFitness += fitness * 3;
                processedNewInBatch+=3;
            }
            accumulatedFitness += fitness;
            processedInBatch++;
        }

        if (tokenizer != null)
            generatedText = tokenizer.Detokenize(outputTokens);

        generationComplete = true;
        go = false;
    }

    private void ResetNeurons()
    {
        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;
    }


    private void ComputeExpectedToken()
    {
        if (targetTokens != null && targetTokens.Count > 2)
        {
            int seedCount = seedTokens.Count;
            // targetTokens: [SOS, token1, token2, ..., tokenN, EOS]
            // ожидаемый токен идёт сразу после seed
            if (seedCount + 1 < targetTokens.Count)
            {
                expectedTokenId = targetTokens[seedCount + 1];
            }
            else
            {
                expectedTokenId = Tokenizer.EOS_TOKEN; // fallback
            }
        }
        else
        {
            expectedTokenId = Tokenizer.UNK_TOKEN;
        }
    }

    // Вспомогательный метод для прямого прохода
    private void RunNetwork()
    {
        for (int i = 0; i < order.Count; i++)
        {
            int thisNeuron = order[i];
            // Применяем активацию ко всем нейронам, кроме входных (индексы 0,1 и входные токены)
            if (thisNeuron >= 2 + vocabSize) // всё, что после входных токенов
            {
                neurones[thisNeuron] = (float)activationFunction(neurones[thisNeuron]);
            }
            // Распространение сигнала по исходящим feed‑forward связям
            if (adjList.Count > thisNeuron)
            {
                foreach (var b in adjList[thisNeuron])
                    if (b.Key < neurones.Count)
                        neurones[b.Key] += b.Value * neurones[thisNeuron];
            }
        }
    }
    
    private int GetPredictedToken()
    {
        float maxActivation = -999f;
        int predictedToken = Tokenizer.UNK_TOKEN;
        
        int outputStart = outConnections;
        int outputEnd = Mathf.Min(outConnections + vocabSize, neurones.Count);
        
        for (int i = outputStart; i < outputEnd; i++)
        {
            float activation = neurones[i];
            if (activation > maxActivation)
            {
                maxActivation = activation;
                predictedToken = i - outConnections;
            }
        }
        
        if (maxActivation <= -0.9f || predictedToken < 0 || predictedToken >= vocabSize)
            return Tokenizer.UNK_TOKEN;
        
        return predictedToken;
    }
    
    private void UpdateRNNState()
    {
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (actConnect[i] == true && RNNs[i] == true)
            {
                if (outInnov[i] < RNNneurones.Count)
                {
                    RNNneurones[outInnov[i]] += neurones[inpInnov[i]] * weights[i];
                }
            }
        }
        
        for (int i = 0; i < Mathf.Min(RNNneurones.Count, neurones.Count); i++)
        {
            neurones[i] = RNNneurones[i];
        }
        
        for (int i = 0; i < RNNneurones.Count; i++)
        {
            RNNneurones[i] *= 0.95f;
        }
    }
    
    public void SetTokenizer(Tokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = tokenizer.GetVocabSize();
        // Синхронизируем outConnections с логикой Start(): выходные нейроны начинаются после входных (bias + vocabSize входов)
        outConnections = 2 + vocabSize;
    }
    
    public void UpdateAvailableTokens(HashSet<int> tokens)
    {
        availableTokens = new HashSet<int>(tokens);
        UpdateAvailableNeurons(availableTokens);
    }
    
    public void AddNode()
    {
        int ind = UnityEngine.Random.Range(0, outInnov.Count);
        int reccurency = 0;
        while (reccurency < 5)
        {
            if (RNNs[ind] == false && actConnect[ind] == true)
            {
                break;
            }
            ind = UnityEngine.Random.Range(0, outInnov.Count);
            ++reccurency;
        }
        if (reccurency >= 5)
        {
            return;
        }
        neurones.Add(0);
        adjList.Add(new Dictionary<int, float>());

        actConnect[ind] = false;

        weights.Add(weights[ind]);
        inpInnov.Add(inpInnov[ind]);
        outInnov.Add(neurones.Count - 1);
        RNNs.Add(false);
        actConnect.Add(true);
        RNNneurones.Add(0);
        innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));

        weights.Add(1f);
        inpInnov.Add(neurones.Count - 1);
        outInnov.Add(outInnov[ind]);
        RNNs.Add(false);
        actConnect.Add(true);
        innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
        makeOrder();
    }
    
    public void AddLink()
    {
        bool errorInOut = false;
        weights.Add(UnityEngine.Random.Range(-3f, 3f));
        
        if (availableInputNeurons.Count == 0)
        {
            Debug.Log("No available input neurons!");
            return;
        }
        inpInnov.Add(availableInputNeurons[UnityEngine.Random.Range(0, availableInputNeurons.Count)]);
        
        List<int> TakenConnections = new List<int>();
        
        if (availableOutputNeurons.Count == 0)
        {
            Debug.Log("No available output neurons!");
            return;
        }
        int probableOut = availableOutputNeurons[UnityEngine.Random.Range(0, availableOutputNeurons.Count)];

        for (int i = 0; i < outInnov.Count; i++)
        {
            if (inpInnov[i] == inpInnov[inpInnov.Count - 1])
            {
                TakenConnections.Add(outInnov[i]);
            }
        }
        
        foreach (int a in TakenConnections)
        {
            if (a == probableOut || probableOut == inpInnov.Last())
            {
                errorInOut = true;
                break;
            }
        }
        
        if (probableOut >= neurones.Count)
        {
            Debug.Log("!!!");
            probableOut = neurones.Count - 1;
        }
        
        outInnov.Add(probableOut);
        RNNs.Add(false);
        actConnect.Add(true);
        
        if (errorInOut == true || GenToPh().SequenceEqual(new List<int> { 1, outConnections, 0 }))
        {
            inpInnov.RemoveAt(inpInnov.Count - 1);
            outInnov.RemoveAt(outInnov.Count - 1);
            RNNs.RemoveAt(RNNs.Count - 1);
            actConnect.RemoveAt(actConnect.Count - 1);
            weights.RemoveAt(weights.Count - 1);
            makeOrder();
            ++recursionAddLink;
            if (recursionAddLink < 3)
            {
                AddLink();
                recursionAddLink = 0;
            }
        }
        else
        {
            if (UnityEngine.Random.Range(0, 6) >= 3)
            {
                if (RNNs.Count > 0)
                {
                    RNNs[RNNs.Count - 1] = true;
                    innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
                }
            }
            else
            {
                innovations.Add(spawnerOfNN.GetComponent<GameManager>().DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
            }
        }
        makeOrder();
    }
    
    public List<int> GenToPh()
    {
        // Копируем данные
        List<float> _weights = new List<float>(weights);
        List<int> _inpInnov = new List<int>(inpInnov);
        List<int> _outInnov = new List<int>(outInnov);
        List<bool> _actConnect = new List<bool>(actConnect);
        List<bool> _RNNs = new List<bool>(RNNs);

        // --- 1. Множество всех нейронов, реально используемых в любой enabled связи ---
        HashSet<int> usedNeurons = new HashSet<int>();
        usedNeurons.Add(0); // bias

        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i]) // связь активна
            {
                usedNeurons.Add(_inpInnov[i]);
                usedNeurons.Add(_outInnov[i]);
            }
        }

        // --- 2. Построение adjList ТОЛЬКО для feed-forward активных связей ---
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
            adjList.Add(new Dictionary<int, float>());

        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i] && !_RNNs[i]) // feed-forward, enabled
            {
                adjList[_inpInnov[i]][_outInnov[i]] = _weights[i];
            }
        }

        // --- 3. Степени входа (in-degree) только для feed-forward связей и только usedNeurons ---
        Dictionary<int, int> inDegree = new Dictionary<int, int>();
        foreach (int n in usedNeurons)
            inDegree[n] = 0;

        for (int i = 0; i < neurones.Count; i++)
        {
            if (!usedNeurons.Contains(i)) continue;
            foreach (var kv in adjList[i]) // kv = KeyValuePair<int, float>
            {
                int to = kv.Key;
                if (usedNeurons.Contains(to))
                    inDegree[to]++;
            }
        }

        // --- 4. Топологическая сортировка ---
        List<int> nullConn = new List<int>();
        foreach (int n in usedNeurons)
            if (inDegree[n] == 0)
                nullConn.Add(n);

        List<int> order1 = new List<int>();
        for (int i = 0; i < nullConn.Count; i++)
        {
            int u = nullConn[i];
            order1.Add(u);
            foreach (var kv in adjList[u])
            {
                int v = kv.Key;
                if (usedNeurons.Contains(v))
                {
                    inDegree[v]--;
                    if (inDegree[v] == 0)
                        nullConn.Add(v);
                }
            }
        }

        // Проверка на циклы
        if (order1.Count != usedNeurons.Count)
        {
            return new List<int> { 1, outConnections, 0 }; // сигнал ошибки
        }

        return order1;
    }
    
    public void makeOrder()
    {
        order = GenToPh();
        // Сохраняем информацию о наличии feed-forward входов
        neuronHasFFIncoming.Clear();
        // Сначала у всех false
        foreach (int n in order)
            neuronHasFFIncoming[n] = false;
        // Проходим по feed-forward связям и отмечаем цели
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (actConnect[i] && !RNNs[i]) // enabled и не рекуррентная
            {
                int to = outInnov[i];
                if (neuronHasFFIncoming.ContainsKey(to))
                    neuronHasFFIncoming[to] = true;
            }
        }
    }
    
    public void Adder()
    {
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
        {
            adjList.Add(new Dictionary<int, float>());
        }
        if (addition == 1)
        {
            AddLink();
        }
        if (addition == 2)
        {
            AddNode();
        }
        makeOrder();
        addition = 0;
    }

    public bool correctGen(List<int> a)
    {
        if (a == null || a.Count == 0) return false;
        if (a.SequenceEqual(new List<int> { 1, outConnections, 0 })) return false;
        return true;
    }

    public void ResetForNewPhase()
    {
        go = true;
        generationComplete = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        generatedText = "";
        outputTokens.Clear();
        
        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;
        
        UpdateAvailableNeurons(availableTokens);

        // Пересоздаём seedTokens на основе текущего inputPhrase
        if (tokenizer != null && targetTokens != null && targetTokens.Count > 2)
        {
            int maxSeed = Mathf.Min(seedLength, targetTokens.Count - 2);
            seedTokens = targetTokens.Skip(1).Take(maxSeed).ToList();
            if (seedTokens.Count == 0)
                seedTokens.Add(Tokenizer.SOS_TOKEN);
        }
        else
        {
            seedTokens = new List<int>() { Tokenizer.SOS_TOKEN };
        }

        // NEW: пересчитываем ожидаемый токен
        ComputeExpectedToken();
        
        seedProcessed = false;
        generationComplete = false;
        seedIndex = 0;
        currentContext.Clear();
        outputTokens.Clear();
    }
    public void UpdateAvailableNeurons(HashSet<int> activeTokens)
    {
        availableInputNeurons.Clear();
        availableOutputNeurons.Clear();

        // Bias всегда доступен
        availableInputNeurons.Add(0);
        // Phase всегда доступен
        availableInputNeurons.Add(1);

        // Входные токены (индексы 2..1+vocabSize)
        foreach (int token in activeTokens)
        {
            int inputIdx = token + 2;
            if (inputIdx < neurones.Count)
                availableInputNeurons.Add(inputIdx);
        }

        // Выходные токены (начинаются с outConnections)
        foreach (int token in activeTokens)
        {
            int outputIdx = outConnections + token;
            if (outputIdx < neurones.Count)
                availableOutputNeurons.Add(outputIdx);
        }

        // Скрытые нейроны (после выходных)
        int hiddenStart = outConnections + vocabSize;
        for (int i = hiddenStart; i < neurones.Count; i++)
        {
            availableInputNeurons.Add(i);
            availableOutputNeurons.Add(i);
        }
    }

    public NeatModelData ToModelData()
    {
        var data = new NeatModelData();
        data.vocabulary = tokenizer.ExportVocabulary();
        data.vocabSize = vocabSize;
        data.outputStart = outConnections;

        // Размер сети
        data.neuronCount = neurones.Count;

        // Топологический порядок и флаги activationFunction
        data.order = order.ToArray();
        data.neuronHasFFIncoming = new bool[neurones.Count];
        foreach (var kv in neuronHasFFIncoming)
            data.neuronHasFFIncoming[kv.Key] = kv.Value;

        // RNN decay
        data.rnnDecay = 0.95f;

        // Связи
        data.connections = new ConnectionGene[inpInnov.Count];
        for (int i = 0; i < inpInnov.Count; i++)
        {
            data.connections[i] = new ConnectionGene
            {
                from = inpInnov[i],
                to = outInnov[i],
                weight = weights[i],
                enabled = actConnect[i],
                recurrent = RNNs[i]
            };
        }

        return data;
    }
    public float GetBatchAverageFitness()
    {
        if (processedInBatch == 0)
        {
            textHowIsGood = 0f;
            return 0f;
        }
        textHowIsGood = accumulatedFitness / processedInBatch;
        return textHowIsGood;
    }

    public float GetNewPhraseAverageFitness()
    {
        if (processedNewInBatch == 0)
        {
            newPhraseFitness = 0f;
            return 0f;
        }
        newPhraseFitness = accumulatedNewFitness / processedNewInBatch;
        return newPhraseFitness;
    }
    private float[] Softmax(float[] activations, float temperature = 1.0f)
    {
        float[] exp = new float[activations.Length];
        float sum = 0f;
        for (int i = 0; i < activations.Length; i++)
        {
            exp[i] = Mathf.Exp(activations[i] / temperature);
            sum += exp[i];
        }
        for (int i = 0; i < activations.Length; i++)
            exp[i] /= sum;
        return exp;
    }

    public double activationFunction(double x)
    {
        //ReLU or Tanh or GelU or something else?
        return Math.Tanh(x);
    }
    
    public void ResetBatchStats()
    {
        accumulatedFitness = 0f;
        processedInBatch = 0;
        accumulatedNewFitness = 0f;
        processedNewInBatch = 0;
    }

    public int getInitalNeurones()
    {
        return initalNeurones;
    }

    public int getOutConnections()
    {
        return outConnections;
    }
}
