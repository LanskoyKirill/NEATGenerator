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
    public int maxIterations = 1000;
    private int iterations = 0;

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
    private int outConnections = 1; // Только bias на входе
    private int initalNeurones;
    public int desiredNeurones;

    // Состояние
    private bool isGenerating = false;
    private int currentTokenIndex = 0;
    private int nextInputToken = 1; // SOS token
    
    // Токенизатор
    [NonSerialized] public Tokenizer tokenizer;
    [NonSerialized] public int vocabSize;
    
    private HashSet<int> availableTokens = new HashSet<int>();
    private List<int> availableInputNeurons = new List<int>();
    private List<int> availableOutputNeurons = new List<int>();

    public void Start()
    {
        go = true;
        isGenerating = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        currentTokenIndex = 0;
        nextInputToken = 1;
        generatedText = "";
        outputTokens.Clear();

        // 1. bias нейрон (0)
        // 2. входные нейроны для токенов (1..vocabSize)
        // 3. выходные нейроны (vocabSize+1..vocabSize*2)
        // 4. скрытые нейроны (после vocabSize*2)
        
        initalNeurones = 1 + vocabSize + vocabSize; // bias + входы + выходы
        
        if (neurones.Count < initalNeurones)
        {
            neurones.Clear();
            RNNneurones.Clear();
            
            // bias
            neurones.Add(1f);
            RNNneurones.Add(1f);
            
            // входные нейроны для токенов
            for (int i = 0; i < vocabSize; i++)
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }
            
            // выходные нейроны
            for (int i = 0; i < vocabSize; i++)
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }
        }
        else
        {
            // Сбрасываем все нейроны
            for (int i = 0; i < neurones.Count; i++)
            {
                neurones[i] = 0f;
                RNNneurones[i] = 0f;
            }
            neurones[0] = 1f; // bias
            RNNneurones[0] = 1f;
        }
        
        // Теперь выходные нейроны начинаются после входных
        outConnections = 1 + vocabSize;
        
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

        UpdateAvailableNeurons();
        
        makeOrder();
        Adder();
        
        // Загрузка материала
        Material AIMaterial = Resources.Load(spawnerOfNN.name, typeof(Material)) as Material;
        if (AIMaterial != null)
        {
            gameObject.GetComponent<Renderer>().material = AIMaterial;
        }
        
        if (order.Count > 3 && order[0] == 1 && order[1] == 0 && order[2] == 0)
        {
            ++testing;
        }
        
        // Токенизация если токенизатор есть
        if (tokenizer != null && !string.IsNullOrEmpty(inputPhrase) && !string.IsNullOrEmpty(targetPhrase))
        {
            inputTokens = tokenizer.Tokenize(inputPhrase);
            targetTokens = tokenizer.Tokenize(targetPhrase);
        }
    }
    
    void Update()
    {
        if (!inpInnov.Any())
        {
            Destroy(gameObject);
            return;
        }
        
        if (go == true)
        {
            neurones[0] = 1;
            
            if (tokenizer == null || targetTokens.Count == 0)
            {
                Debug.Log("No tokenizer or targetTokens is empty!");
                return;
            }
            
            ProcessGenerationLogic();
        }
    }
    
    private void ProcessGenerationLogic()
    {
        // Сброс нейронов кроме bias
        for (int i = 1; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
        }
        neurones[0] = 1f;
        
        // Генерация начинается с SOS
        if (!isGenerating)
        {
            nextInputToken = Tokenizer.SOS_TOKEN;
            isGenerating = true;
            outputTokens.Clear();
            outputTokens.Add(Tokenizer.SOS_TOKEN);
        }
        
        // Подаем текущий токен на вход (входные нейроны 1..vocabSize)
        int currentToken = nextInputToken;
        if (currentToken >= 0 && currentToken < vocabSize)
        {
            // Входной нейрон для токена: 1 + token
            int inputNeuronIndex = 1 + currentToken;
            if (inputNeuronIndex < neurones.Count && inputNeuronIndex >= 1)
            {
                neurones[inputNeuronIndex] = 1f;
            }
            else
            {
                Debug.LogError($"Input neuron index out of range: {inputNeuronIndex}, neurones count: {neurones.Count}");
            }
        }
        
        // Пропускаем через сеть
        for (int i = 0; i < order.Count; i++)
        {
            int thisNeuron = order[i];
            if (thisNeuron >= 1) // Все нейроны кроме bias используют активацию
            {
                neurones[thisNeuron] = (float)Math.Tanh(neurones[thisNeuron]);
            }
            if (adjList.Count > thisNeuron)
            {
                foreach (var b in adjList[thisNeuron])
                {
                    if (b.Key < neurones.Count)
                    {
                        neurones[b.Key] += b.Value * neurones[thisNeuron];
                    }
                }
            }
        }
        
        // Генерация следующего токена (выходные нейроны начинаются с outConnections)
        float maxActivation = -999f;
        int predictedToken = Tokenizer.UNK_TOKEN;
        
        int outputStart = outConnections;
        int outputEnd = Mathf.Min(outConnections + vocabSize, neurones.Count);
        
        //Debug.Log($"Checking output neurons from {outputStart} to {outputEnd}");
        
        for (int i = outputStart; i < outputEnd; i++)
        {
            //Debug.Log($"Output neuron {i} activation: {neurones[i]}");
            if (neurones[i] > maxActivation)
            {
                maxActivation = neurones[i];
                predictedToken = i - outConnections;
                //Debug.Log($"New best token: {predictedToken} with activation {maxActivation}");
            }
        }
        
        // Если все активации отрицательные, используем UNK
        if (maxActivation <= -1f)
        {
            predictedToken = Tokenizer.UNK_TOKEN;
        }
        
        // Проверяем, что токен в допустимом диапазоне
        if (predictedToken < 0 || predictedToken >= vocabSize)
        {
            predictedToken = Tokenizer.UNK_TOKEN;
        }
        
        // Сохраняем токен (кроме повторного SOS)
        if (!(outputTokens.Count > 0 && outputTokens.Last() == Tokenizer.SOS_TOKEN && predictedToken == Tokenizer.SOS_TOKEN))
        {
            outputTokens.Add(predictedToken);
            //Debug.Log($"Added token {predictedToken} to output. Total tokens: {outputTokens.Count}");
        }
        
        // Детокенизируем для отображения
        if (tokenizer != null)
        {
            generatedText = tokenizer.Detokenize(outputTokens);
            //Debug.Log($"Generated text: {generatedText}");
        }
        
        // Подготавливаем следующий вход (авторегрессия)
        nextInputToken = predictedToken;
        
        // Проверка окончания генерации
        bool shouldFinish = false;
        
        if (predictedToken == Tokenizer.EOS_TOKEN)
        {
            shouldFinish = true;
        }
        else if (outputTokens.Count >= Math.Max(targetTokens.Count + 10, 50))
        {
            //Debug.Log($"Max length reached ({outputTokens.Count}), finishing generation");
            shouldFinish = true;
        }
        else if (iterations >= maxIterations)
        {
            //Debug.Log($"Max iterations reached ({iterations}), finishing generation");
            shouldFinish = true;
        }
        
        if (shouldFinish)
        {
            FinishGeneration();
        }
        
        UpdateRNNState();
        iterations++;
    }
    
    private void FinishGeneration()
    {
        go = false;
        isGenerating = false;
        
        CalculateTokenFitness();
        
        cIterator = 0;
    }
    
    private void CalculateTokenFitness()
    {
        if (targetTokens.Count == 0 || outputTokens.Count == 0)
        {
            howIsGood = 0f;
            return;
        }
        
        // 1. Точное совпадение последовательностей
        int exactMatches = 0;
        int minLength = Mathf.Min(outputTokens.Count, targetTokens.Count);
        
        for (int i = 0; i < minLength; i++)
        {
            if (outputTokens[i] == targetTokens[i])
                exactMatches++;
        }
        
        float exactMatchScore = targetTokens.Count > 0 ? 
            (float)exactMatches / targetTokens.Count : 0f;
        
        // 2. Совпадение уникальных слов (исключая служебные)
        var outputWords = new HashSet<int>();
        var targetWords = new HashSet<int>();
        
        foreach (var token in outputTokens)
        {
            if (token > 3) // Исключаем служебные
                outputWords.Add(token);
        }
        
        foreach (var token in targetTokens)
        {
            if (token > 3)
                targetWords.Add(token);
        }
        
        outputWords.IntersectWith(targetWords);
        
        float wordMatchScore = targetWords.Count > 0 ? 
            (float)outputWords.Count / targetWords.Count : 0f;
        
        // 3. Наибольшая общая подпоследовательность
        float lcsScore = CalculateLCS(outputTokens, targetTokens);
        
        // 4. Бонус за правильную структуру
        float structureBonus = 0f;
        
        if (outputTokens.Count > 0 && outputTokens[0] == Tokenizer.SOS_TOKEN)
            structureBonus += 0.03f;
        
        if (outputTokens.Count > 0 && outputTokens.Last() == Tokenizer.EOS_TOKEN)
            structureBonus += 0.03f;
        
        // 5. Штраф за слишком длинную генерацию
        float penaltyForLength = 0;
        if (outputTokens.Count > targetTokens.Count * 1.5f)
        {
            penaltyForLength = -(outputTokens.Count / (float)targetTokens.Count) * 0.1f;
        }
        
        // 6. Бонус за разнообразие (использование разных слов)
        float diversityBonus = 0f;
        if (outputWords.Count > 3)
        {
            diversityBonus = Mathf.Min(0.1f, outputWords.Count / 30f);
        }
        
        // 7. Штраф за повторение одного слова много раз
        float repetitionPenalty = 0f;
        var tokenCounts = new Dictionary<int, int>();
        foreach (var token in outputTokens)
        {
            if (token > 3)
            {
                if (!tokenCounts.ContainsKey(token)) tokenCounts[token] = 0;
                tokenCounts[token]++;
            }
        }
        foreach (var kvp in tokenCounts)
        {
            if (kvp.Value > 5)
            {
                repetitionPenalty -= 0.05f;
            }
        }
        
        textHowIsGood = Mathf.Min(200.0f, 
            exactMatchScore * 0.4f + 
            wordMatchScore * 0.3f + 
            lcsScore * 0.2f +
            structureBonus +
            penaltyForLength +
            diversityBonus +
            repetitionPenalty);
        
        howIsGood = Mathf.Min(200.0f, textHowIsGood);
        
        accumulatedFitness += howIsGood;
        ++processedInBatch;
    }
    
    private float CalculateLCS(List<int> a, List<int> b)
    {
        if (a.Count == 0 || b.Count == 0) return 0f;
        
        int[,] dp = new int[a.Count + 1, b.Count + 1];
        
        for (int i = 1; i <= a.Count; i++)
        {
            for (int j = 1; j <= b.Count; j++)
            {
                if (a[i - 1] == b[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                }
                else
                {
                    dp[i, j] = Mathf.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }
        }
        
        return (float)dp[a.Count, b.Count] / Mathf.Max(a.Count, b.Count, 1);
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
        outConnections = 1; // Только bias для генератора
    }
    
    public void UpdateAvailableTokens(HashSet<int> tokens)
    {
        availableTokens = new HashSet<int>(tokens);
        UpdateAvailableNeurons();
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
            if (UnityEngine.Random.Range(0, 6) >= 2)
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
        List<float> _weights = new List<float>(weights);
        List<int> _inpInnov = new List<int>(inpInnov);
        List<int> _outInnov = new List<int>(outInnov);
        List<bool> _actConnect = new List<bool>(actConnect);
        List<bool> _RNNs = new List<bool>(RNNs);
        List<int> nullConn = new List<int>();
        List<int> inDegree = new List<int>();
        List<int> order1 = new List<int>();
        int neuronesCount = neurones.Count;
        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i] == false || _RNNs[i] == true)
            {
                _actConnect.RemoveAt(i);
                _weights.RemoveAt(i);
                _inpInnov.RemoveAt(i);
                _outInnov.RemoveAt(i);
                _RNNs.RemoveAt(i);
                --i;
            }
        }
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
        {
            adjList.Add(new Dictionary<int, float>());
        }
        for (int i = 0; i < _inpInnov.Count; i++)
        {
            adjList[_inpInnov[i]].Add(_outInnov[i], _weights[i]);
        }
        for (int i = 0; i < adjList.Count; i++)
        {
            inDegree.Add(0);
        }
        for (int i = 0; i < neurones.Count; i++)
        {
            foreach (var b in adjList[i])
            {
                ++inDegree[b.Key];
            }
        }
        for (int i = 0; i < inDegree.Count; i++)
        {
            if (inDegree[i] == 0)
            {
                nullConn.Add(i);
            }
        }
        for (int i = 0; i != nullConn.Count; ++i)
        {
            int ie = nullConn[i];
            order1.Add(ie);
            foreach (var b in adjList[ie])
            {
                int a = b.Key;
                --inDegree[b.Key];
                if (inDegree[a] == 0)
                {
                    nullConn.Add(b.Key);
                }
            }
        }
        if (order1.Count != neuronesCount)
        {
            return new List<int> { 1, outConnections, 0 };
        }
        if (!order1.Any())
        {
            Debug.Log("Empty");
            return new List<int> { 1, outConnections, 0 };
        }
        return order1;
    }
    
    public void makeOrder()
    {
        List<int> a = GenToPh();
        order = a;
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
        List<int> genes = new List<int>(a);
        bool yes = true;
        if(a.SequenceEqual(new List<int> {1, outConnections, 0})){
            return false;
        }
        if (genes[0] == 0 || genes[0] > outConnections)
        {
            genes.RemoveRange(0, genes.Count - 3);
            for (int i = 1; i != outConnections + 1; ++i)
            {
                if (genes.Contains(i))
                {
                    yes = false;
                    break;
                }
            }
        }
        return yes;
    }

    public void ResetForNewPhase()
    {
        go = true;
        isGenerating = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        currentTokenIndex = 0;
        nextInputToken = 1;
        generatedText = "";
        outputTokens.Clear();
        iterations = 0;
        
        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;
        
        UpdateAvailableNeurons();
    }

    private void UpdateAvailableNeurons()
    {
        availableInputNeurons.Clear();
        availableOutputNeurons.Clear();
        
        // Входные нейроны:
        // 1. bias (0)
        availableInputNeurons.Add(0);
        
        // 2. Все входные нейроны для токенов (1..vocabSize)
        for (int i = 1; i <= vocabSize; i++)
        {
            if (i < neurones.Count)
            {
                availableInputNeurons.Add(i);
            }
        }
        
        // Выходные нейроны:
        // Начинаются с outConnections (1 + vocabSize)
        for (int i = outConnections; i < outConnections + vocabSize; i++)
        {
            if (i < neurones.Count)
            {
                availableOutputNeurons.Add(i);
            }
        }
        
        // Скрытые нейроны (если есть)
        int hiddenStart = outConnections + vocabSize;
        for (int i = hiddenStart; i < neurones.Count; i++)
        {
            // Скрытые нейроны могут быть и входами и выходами
            if (!availableInputNeurons.Contains(i))
                availableInputNeurons.Add(i);
            if (!availableOutputNeurons.Contains(i))
                availableOutputNeurons.Add(i);
        }
    }

    public float GetBatchAverageFitness()
    {
        if (processedInBatch == 0) return 0f;
        return accumulatedFitness / processedInBatch;
    }
    
    public void ResetBatchStats()
    {
        accumulatedFitness = 0f;
        processedInBatch = 0;
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