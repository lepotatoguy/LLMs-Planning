### Step 1: Make sure you're on a Linux machine with Python 3.6+
(I use Python 3.10)
`conda create -n planbench python=3.10`
`conda activate planbench`

### Step 2: Install Python dependencies
`pip install -r requirements.txt`

### Step 3: Set up Fast Downward (planner)
## Download from: https://github.com/aibasel/downward
## Then set the environment variable
`export FAST_DOWNWARD=/path/to/fast_downward`

/home/joyanta/downward

### Step 4: Set up VAL (validator)
## Git clone from: https://github.com/KCL-Planning/VAL
## Or use from planner_tools [LLMs-Planning/planner_tools/VAL]. (Recommended)
`export VAL=/path/to/val`
export VAL=/home/joyanta/Study/Research/PlanBench-OG/LLMs-Planning/planner_tools/VAL


### Step 5: Set up PR2Plan (obs-compiler)
## Git clone from: https://sites.google.com/site/prasplanning/file-cabinet
## Or use from planner_tools [LLMs-Planning/planner_tools/PR2]. (Recommended)
`export PR2=/path/to/pr2plan`
export PR2=/home/joyanta/Study/Research/PlanBench-OG/LLMs-Planning/planner_tools/PR2

### Step 6 (Optional): Set up BLOOM (if you're using BLOOM instead of OpenAI)
export BLOOM_CACHE_DIR=/path/to/bloom/cache/dir

### Step 6.1 Create OpenAI API Key if you haven't, and if you plan to use anything like this

https://platform.openai.com/api-keys

### Step 6.2: For something opensource, use OLLAMA. https://ollama.com/ (Recommended)
Download, then install.
Then,
`ollama pull qwen:7b`

### Step 7: Integrate the API in `utils/llm_utils.py` [Line 56]

### Step 8: Run the full PlanBench pipeline
`python3 llm_plan_pipeline.py --task t1 --config logistics --engine ollama`
or 
`python3 llm_plan_pipeline.py --task t1 --config logistics --engine gpt-3.5-turbo_chat`

### Optional: Run only prompt generation
`python3 prompt_generation.py --task t1 --config logistics`

### Optional: Run only response generation (after prompt generation)
`python3 response_generation.py --task t1 --config logistics --engine ollama`

### Optional: Run only evaluation (after response generation)
`python3 response_evaluation.py --task t1 --config logistics --engine ollama`