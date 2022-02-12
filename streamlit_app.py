import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

first = """informal english: corn fields are all across illinois, visible once you leave chicago.\nTranslated into the Style of Abraham Lincoln: corn fields ( permeate illinois / span the state of illinois / ( occupy / persist in ) all corners of illinois / line the horizon of illinois / envelop the landscape of illinois ), manifesting themselves visibly as one ventures beyond chicago.\n\ninformal english:"""

@st.cache(allow_output_mutation=True)
def get_model():
    print("hello")
    model = AutoModelWithLMHead.from_pretrained("BigSalmon/InformalToFormalLincoln21")
    print("hello")
    tokenizer = AutoTokenizer.from_pretrained("BigSalmon/Points2")
    print("hello")
    return model, tokenizer
    
model, tokenizer = get_model()

st.text('''How To Make Prompt:
informal english: sega’s doing mobile gaming right with this approach to exploiting its back catalog, giving people a way to build up their library of classics on devices they’ll always have close to hand for cheap. it’s a far better experience than nintendo’s classic machine sales on its home gaming systems, which tend to be locked in and not passed between hardware generations.
Translated into the Style of Abraham Lincoln: pursuing the proper course, sega's foray into mobile gaming entails leveraging its back catalog of classics on portable devices that ( are certain to be in reach whenever / will be at the ready for use at all times ) ( without spending a fortune / at minimal cost ). such an endeavor comes as a welcome alternative to nintendo's propensity to ( sell / offer / market ) home gaming systems that are rigidly bound to a single hardware generation.
Translated into the Style of Abraham Lincoln: at a time when nintendo has become inflexible, ( stubbornly bent on / firmly set on / unyielding in its insistence on / steadfastly transfixed by / uncompromising in its commitment to / rigidly decided on ) consoles that are tethered to a fixed iteration, sega diligently curates its legacy of classic video games on handheld devices.
informal english: garage band has made people who know nothing about music good at creating music.
Translated into the Style of Abraham Lincoln: garage band ( offers the uninitiated in music the ability to produce professional-quality compositions / catapults those for whom music is an uncharted art the ability the realize masterpieces / stimulates music novice's competency to yield sublime arrangements / begets individuals of rudimentary musical talent the proficiency to fashion elaborate suites ).
informal english: chrome extensions can make doing regular tasks much easier to get done.
Translated into the Style of Abraham Lincoln: chrome extensions ( yield the boon of time-saving convenience / ( expedite the ability to / unlock the means to more readily ) accomplish everyday tasks / turbocharges the velocity with which one can conduct their obligations ).
informal english: broadband is finally expanding to rural areas, a great development that will thrust them into modern life.
Translated into the Style of Abraham Lincoln: broadband is ( ( finally / at last / after years of delay ) arriving in remote locations / springing to life in far-flung outposts / inching into even the most backwater corners of the nation ) that will ( hasten their transition into the modern age / leap-frog them into the twenty-first century / facilitate their integration into contemporary life ).
informal english: national parks are a big part of the us culture.
Translated into the Style of Abraham Lincoln: the culture of the united states is ( inextricably ( bound up with / molded by / enriched by / enlivened by ) its ( serene / picturesque / pristine / breathtaking ) national parks ).
informal english: corn fields are all across illinois, visible once you leave chicago.
Translated into the Style of Abraham Lincoln: corn fields ( permeate illinois / span the state of illinois / ( occupy / persist in ) all corners of illinois / line the horizon of illinois / envelop the landscape of illinois ), manifesting themselves visibly as one ventures beyond chicago.
informal english:''')

temp = st.sidebar.slider("Temperature", 0.7, 1.5)
number_of_outputs = st.sidebar.slider("Number of Outputs", 5, 50)
lengths = st.sidebar.slider("Length", 3, 10)
bad_words = st.text_input("Words You Do Not Want Generated", " core lemon height time ")

def run_generate(text, bad_words):
  yo = []
  input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
  res = len(tokenizer.encode(text))
  bad_words = bad_words.split()
  bad_word_ids = []
  for bad_word in bad_words: 
    bad_word = " " + bad_word
    ids = tokenizer(bad_word).input_ids
    bad_word_ids.append(ids)
  sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length= res + lengths, 
    min_length = res + lengths, 
    top_k=50,
    temperature=temp,
    num_return_sequences=number_of_outputs,
    bad_words_ids=bad_word_ids
  )
  for i in range(number_of_outputs):
    e = tokenizer.decode(sample_outputs[i])
    e = e.replace(text, "")
    yo.append(e)
  return yo
with st.form(key='my_form'):
    text = st.text_area(label='Enter sentence', value=first)
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
      translated_text = run_generate(text, bad_words)
      st.write(translated_text if translated_text else "No translation found")
      with torch.no_grad():
        text2 = str(text)
        print(text2)
        text3 = tokenizer.encode(text2)
        myinput, past_key_values = torch.tensor([text3]), None
        myinput = myinput
        myinput= myinput.to(device)
        logits, past_key_values = model(myinput, past_key_values = past_key_values, return_dict=False)
        logits = logits[0,-1]
        probabilities = torch.nn.functional.softmax(logits)
        best_logits, best_indices = logits.topk(100)
        best_words = [tokenizer.decode([idx.item()]) for idx in best_indices]      
        st.write(best_words)
