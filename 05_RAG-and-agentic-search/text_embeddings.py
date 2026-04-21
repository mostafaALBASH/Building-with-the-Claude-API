# Client Setup
import os
from dotenv import load_dotenv
import voyageai
import re

_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
client = voyageai.Client()

def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)

# Embedding Generation
def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]



def main ():
    with open(os.path.join(_DIR, "report.md"), "r") as f:
        text = f.read()
        chunks = chunk_by_section(text)
        embedding = generate_embedding(chunks[0])
        print("Embedding for first chunk:", embedding)
        """
        Out should look like:
        embedding for first chunk: [-0.05453480780124664, 0.01431055087596178, -0.016921261325478554, 0.0005922440905123949, 0.02136913500726223,
         0.037903621792793274, -0.047186143696308136, 0.0024173229467123747, 0.0025865354109555483, -0.01798488199710846, 0.014020473696291447, 0.048926617950201035, -0.00918582733720541, -0.0037710238248109818, 
         -0.013730393722653389, 0.007010236848145723, 0.03326236084103584, 0.08857070654630661, -0.04737953096628189, -0.010201102122664452, 0.06033638119697571, -0.057628974318504333, 0.06497763842344284, -0.03461606428027153,
          -0.018468346446752548, 0.007638740353286266, 0.007203621789813042, 0.04486551135778427, -0.03248881921172142, -0.014407243579626083, -0.000906496134120971, 0.022142676636576653, 0.01885511912405491, 0.05685543641448021,
           0.012473385781049728, 0.04099779576063156, -0.06188346818089485, -0.020402204245328903, -0.02726740390062332, -0.054921574890613556, 0.010587873868644238, 0.005245590582489967, 0
        """

# uv run -m app.rag.text_embeddings
if __name__ == "__main__":
    main()


