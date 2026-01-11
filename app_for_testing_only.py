
####################################################
## Boilerplate Sample Code. Ignore this code
####################################################

from flask import Flask, request, jsonify
from src.agents import main_workflow1,main_workflow2
from src.config.settings import settings

save_path__bs = settings.BS_DIR
app = Flask(__name__)

# Initialize the research agent system once at startup
agent_system = main_workflow2()

@app.route("/")
def home():
    return jsonify({"message": "LangGraph Research Agent API is running."})


@app.route("/research", methods=["POST"])
def research():

    data = request.get_json(force=True)

    try:
        # Run the research agent
        bs_name = save_path__bs / '''bs_question.txt'''
        with open(bs_name, 'r') as f:
            lines = f.readlines()
        # Remove newline characters
        l_business_question = [line.strip() for line in lines if len(line.strip()) >= 10]

        for _bs in l_business_question:
            # Try for 3 times. This is due to there could be a bug due to plot generation.
            for attempt in range(3):
                try:
                    main_workflow1(_bs)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
        main_workflow2()
        print('Report Generation Successful!')
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
