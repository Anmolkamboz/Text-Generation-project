from django.shortcuts import render
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

savedModel = load_model('mymodel.keras')

with open('tokenizer.pkl', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

def generate_text(request):
    seed_text = ""
    next_words = 100
    generated_text = "" 

    if request.method == 'POST':
        seed_text = request.POST.get('seed_text', '')
        next_words = int(request.POST.get('next_words', 100))
        generated_text = seed_text

        for _ in range(next_words):
            sequence = loaded_tokenizer.texts_to_sequences([generated_text])
            padded = pad_sequences(sequence, maxlen=19)

            predicted = savedModel.predict(padded, verbose=0)
            predicted_class = predicted.argmax(axis=-1)

            output_word = ''
            for word, index in loaded_tokenizer.word_index.items():
                if index == predicted_class:
                    output_word = word
                    break

            generated_text += ' ' + output_word

    return render(request, 'textgen/generate_text.html', {'seed_text': seed_text, 'generated_text': generated_text})
#python manage.py runserver