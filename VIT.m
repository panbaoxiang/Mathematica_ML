dim={32,32};
div={4,4};
linear=60;
channel=33;
each=dim/div;

preprocessing=NetGraph[<|
			"divide"->{ReshapeLayer[{Automatic,div[[1]],each[[1]],div[[2]],each[[2]]}],TransposeLayer[{3<->4}],FlattenLayer[{{2,3}}],NetMapOperator[linear]},
			"position"->NetArrayLayer["Output" -> {div/.List->Times,linear}],
		    "total"->TotalLayer[]|>,
{NetPort["Input"]->"divide"->"total","position"->"total"},
"Input"->NetEncoder[{"Image",dim}]]

multiheadself = NetInitialize@
  NetGraph[<|"Key" -> NetMapOperator[{channel, linear}], 
    "Value" -> NetMapOperator[{channel, linear}], 
    "Query" -> NetMapOperator[{channel, linear}], 
    "Attention" -> AttentionLayer["Dot", "MultiHead" -> True], 
    "Merge" -> NetMapOperator[linear]|>, 
    {"Key" -> NetPort["Attention", "Key"], 
    "Value" -> NetPort["Attention", "Value"], 
    "Query" -> NetPort["Attention", "Query"],
    "Attention" -> "Merge"}, "Input" -> {"Varying", linear}];

attention=NetGraph[<|"LN1"->NormalizationLayer[],
					"multiheadself"->multiheadself,
					"Add1"->TotalLayer[],
					"LN2"->NormalizationLayer[],
					"Linear"->NetMapOperator[{LinearLayer[linear],Ramp}],
					"Add2"->TotalLayer[]|>,
	{NetPort["Input"]->"LN1"->"multiheadself"->"Add1",
	 NetPort["Input"]->"Add1",
	 "Add1"->"LN2"->"Linear"->"Add2",
	 "Add1"->"Add2"},
	 "Input"->{div/.List->Times,linear}]
   
  vit=NetChain[{preprocessing,attention,attention,attention,FlattenLayer[],10,SoftmaxLayer[]},
	"Input"->NetEncoder[{"Image",dim}],
	"Output"->NetDecoder[{"Class",DeleteDuplicates[training[[;;,2]]]}]]
  
  NetTrain[vit,training,ValidationSet->test,MaxTrainingRounds->500,BatchSize->256]
  
