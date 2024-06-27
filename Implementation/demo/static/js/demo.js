var sources = {};
var generatedSources = {};

async function LoadSelectedPreset() {
    var myform = document.getElementById("DemoForm");
    var preset = myform.elements["presetSelect"].value;

    await LoadPreset(preset);
}

async function LoadPreset(preset) {
    var myform = document.getElementById("DemoForm");
    myform.elements["inverted_source_content"].value = "";
    myform.elements["knowledge_graph"].value = "";
    document.getElementById("subject_count").value = "";
    document.getElementById("triple_count").value = "";
    if (preset === "None") {
        myform.elements["yarrrml"].value = "";
        myform.elements["originalSourceSelect"].innerHTML = "";
        myform.elements["original_source_content"].value = "";
        sources = {};
        return;
    }
    const formData = new FormData();
    formData.set("preset", preset);
    await fetch("/get_preset", {
        method: "POST",
        body: formData,
    }).then(response => response.json())
        .then(data => {
            myform.elements["yarrrml"].value = data["yarrrml"];
            sources = data["sources"];
            SetOriginalSourceSelectOptions();
            myform.elements["originalSourceSelect"].value = Object.keys(sources)[0];
            myform.elements["original_source_content"].value = sources[Object.keys(sources)[0]];
        });
}

function SetOriginalSourceSelectOptions () {
    var sourceSelect = document.getElementById("originalSourceSelect");
    sourceSelect.innerHTML = "";
    for (var source_name in sources) {
        var option = document.createElement("option");
        option.text = source_name;
        option.value = source_name;
        sourceSelect.add(option);
    }
}

function SetGeneratedSourceSelectOptions() {
    var sourceSelect = document.getElementById("invertedSourceSelect");
    sourceSelect.innerHTML = "";
    for (var source_name in generatedSources) {
        var option = document.createElement("option");
        option.text = source_name;
        option.value = source_name;
        sourceSelect.add(option);
    }
}

function SetCurrentOriginalSourceContent(source_name) {
    var myform = document.getElementById("DemoForm");
    myform.elements["original_source_content"].value = sources[source_name];
}

function SetCurrentGeneratedSourceContent(source_name) {
    var myform = document.getElementById("DemoForm");
    myform.elements["inverted_source_content"].value = generatedSources[source_name];
}

async function GenerateKnowledgeGraph() {
    var myform = document.getElementById("DemoForm");
    const formData = new FormData();
    formData.set("yarrrml", myform.elements["yarrrml"].value)
    formData.set("sources", JSON.stringify(sources));
    fetch("/generate_knowledge_graph", {
        method: "POST",
        body: formData,
    }).then(response => response.json())
        .then(data => {
            myform.elements["knowledge_graph"].value = data["graph"];
            document.getElementById("subject_count").innerText = data["subject_count"];
            document.getElementById("triple_count").innerText = data["triple_count"];
        });
}

function ClearKnowledgeGraph() {
    var myform = document.getElementById("DemoForm");
    myform.elements["knowledge_graph"].value = "";
    document.getElementById("subject_count").innerText = "";
    document.getElementById("triple_count").innerText = "";
}

function CopyGeneratedSourcesToSourceContent() {
    sources = generatedSources;
    var myform = document.getElementById("DemoForm");
    var currentSelectedSource = myform.elements["originalSourceSelect"].value;
    if (Object.keys(sources).includes(currentSelectedSource)) {
        myform.elements["original_source_content"].value = sources[currentSelectedSource];
    }
}

function InvertKnowledgeGraph() {
    var myform = document.getElementById("DemoForm");
    const formData = new FormData();
    formData.set("knowledge_graph", myform.elements["knowledge_graph"].value)
    formData.set("yarrrml", myform.elements["yarrrml"].value)
    fetch("/invert_knowledge_graph", {
        method: "POST",
        body: formData,
    }).then(response => response.json())
        .then(data => {
            generatedSources = data["sources"];
            SetGeneratedSourceSelectOptions();
            myform.elements["invertedSourceSelect"].value = Object.keys(generatedSources)[0];
            SetCurrentGeneratedSourceContent(Object.keys(generatedSources)[0]);
        });
}

// onload add event listeners and fetch presets
window.onload = function () {
    fetch("/get_presets")
        .then(response => response.json())
        .then(data => {
            var presetSelect = document.getElementById("presetSelect");
            for (var i = 0; i < data.length; i++) {
                var option = document.createElement("option");
                option.text = data[i];
                option.value = data[i];
                presetSelect.add(option);
            }
        });
    LoadPreset("None");
    console.log("Presets fetched");
    document.getElementById("presetSelect").addEventListener("change", LoadSelectedPreset);
    document.getElementById("reloadPresetButton").addEventListener("click", LoadSelectedPreset);
    document.getElementById("generateKnowledgeGraphButton").addEventListener("click", GenerateKnowledgeGraph);
    document.getElementById("originalSourceSelect").addEventListener("change", function () {
        SetCurrentOriginalSourceContent(this.value);
    });
    document.getElementById("invertedSourceSelect").addEventListener("change", function () {
        SetCurrentGeneratedSourceContent(this.value);
    });
    document.getElementById("clearKnowledgeGraphButton").addEventListener("click", ClearKnowledgeGraph);
    document.getElementById("copyGeneratedSourcesButton").addEventListener("click", CopyGeneratedSourcesToSourceContent);
    document.getElementById("invertKnowledgeGraphButton").addEventListener("click", InvertKnowledgeGraph);
    console.log("Event listeners added");
}