¯£
òÇ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Ñ
|
Adam/fourth/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fourth/bias/v
u
&Adam/fourth/bias/v/Read/ReadVariableOpReadVariableOpAdam/fourth/bias/v*
_output_shapes
:*
dtype0

Adam/fourth/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/fourth/kernel/v
}
(Adam/fourth/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fourth/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/third/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/third/bias/v
s
%Adam/third/bias/v/Read/ReadVariableOpReadVariableOpAdam/third/bias/v*
_output_shapes
:
*
dtype0

Adam/third/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/third/kernel/v
{
'Adam/third/kernel/v/Read/ReadVariableOpReadVariableOpAdam/third/kernel/v*
_output_shapes

:

*
dtype0
|
Adam/second/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/second/bias/v
u
&Adam/second/bias/v/Read/ReadVariableOpReadVariableOpAdam/second/bias/v*
_output_shapes
:
*
dtype0

Adam/second/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/second/kernel/v
}
(Adam/second/kernel/v/Read/ReadVariableOpReadVariableOpAdam/second/kernel/v*
_output_shapes

:

*
dtype0
z
Adam/first/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/first/bias/v
s
%Adam/first/bias/v/Read/ReadVariableOpReadVariableOpAdam/first/bias/v*
_output_shapes
:
*
dtype0

Adam/first/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/first/kernel/v
{
'Adam/first/kernel/v/Read/ReadVariableOpReadVariableOpAdam/first/kernel/v*
_output_shapes

:
*
dtype0
|
Adam/fourth/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/fourth/bias/m
u
&Adam/fourth/bias/m/Read/ReadVariableOpReadVariableOpAdam/fourth/bias/m*
_output_shapes
:*
dtype0

Adam/fourth/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*%
shared_nameAdam/fourth/kernel/m
}
(Adam/fourth/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fourth/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/third/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/third/bias/m
s
%Adam/third/bias/m/Read/ReadVariableOpReadVariableOpAdam/third/bias/m*
_output_shapes
:
*
dtype0

Adam/third/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/third/kernel/m
{
'Adam/third/kernel/m/Read/ReadVariableOpReadVariableOpAdam/third/kernel/m*
_output_shapes

:

*
dtype0
|
Adam/second/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/second/bias/m
u
&Adam/second/bias/m/Read/ReadVariableOpReadVariableOpAdam/second/bias/m*
_output_shapes
:
*
dtype0

Adam/second/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*%
shared_nameAdam/second/kernel/m
}
(Adam/second/kernel/m/Read/ReadVariableOpReadVariableOpAdam/second/kernel/m*
_output_shapes

:

*
dtype0
z
Adam/first/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/first/bias/m
s
%Adam/first/bias/m/Read/ReadVariableOpReadVariableOpAdam/first/bias/m*
_output_shapes
:
*
dtype0

Adam/first/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/first/kernel/m
{
'Adam/first/kernel/m/Read/ReadVariableOpReadVariableOpAdam/first/kernel/m*
_output_shapes

:
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
fourth/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefourth/bias
g
fourth/bias/Read/ReadVariableOpReadVariableOpfourth/bias*
_output_shapes
:*
dtype0
v
fourth/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namefourth/kernel
o
!fourth/kernel/Read/ReadVariableOpReadVariableOpfourth/kernel*
_output_shapes

:
*
dtype0
l

third/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
third/bias
e
third/bias/Read/ReadVariableOpReadVariableOp
third/bias*
_output_shapes
:
*
dtype0
t
third/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namethird/kernel
m
 third/kernel/Read/ReadVariableOpReadVariableOpthird/kernel*
_output_shapes

:

*
dtype0
n
second/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namesecond/bias
g
second/bias/Read/ReadVariableOpReadVariableOpsecond/bias*
_output_shapes
:
*
dtype0
v
second/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namesecond/kernel
o
!second/kernel/Read/ReadVariableOpReadVariableOpsecond/kernel*
_output_shapes

:

*
dtype0
l

first/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
first/bias
e
first/bias/Read/ReadVariableOpReadVariableOp
first/bias*
_output_shapes
:
*
dtype0
t
first/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namefirst/kernel
m
 first/kernel/Read/ReadVariableOpReadVariableOpfirst/kernel*
_output_shapes

:
*
dtype0

NoOpNoOp
û6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶6
value¬6B©6 B¢6
ò
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
¦
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
<
0
1
2
3
%4
&5
-6
.7*
<
0
1
2
3
%4
&5
-6
.7*
* 
°
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
Ô
<iter

=beta_1

>beta_2
	?decay
@learning_ratemcmdmemf%mg&mh-mi.mjvkvlvmvn%vo&vp-vq.vr*
* 

Aserving_default* 

0
1*

0
1*
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
\V
VARIABLE_VALUEfirst/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
first/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
]W
VARIABLE_VALUEsecond/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEsecond/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 
\V
VARIABLE_VALUEthird/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
third/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 

Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
]W
VARIABLE_VALUEfourth/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEfourth/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

^0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
_	variables
`	keras_api
	atotal
	bcount*

a0
b1*

_	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/first/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/first/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/second/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/second/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/third/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/third/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/fourth/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/fourth/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/first/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/first/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/second/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/second/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/third/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/third/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/fourth/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/fourth/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
serving_default_first_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_first_inputfirst/kernel
first/biassecond/kernelsecond/biasthird/kernel
third/biasfourth/kernelfourth/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1376635
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
®
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename first/kernel/Read/ReadVariableOpfirst/bias/Read/ReadVariableOp!second/kernel/Read/ReadVariableOpsecond/bias/Read/ReadVariableOp third/kernel/Read/ReadVariableOpthird/bias/Read/ReadVariableOp!fourth/kernel/Read/ReadVariableOpfourth/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/first/kernel/m/Read/ReadVariableOp%Adam/first/bias/m/Read/ReadVariableOp(Adam/second/kernel/m/Read/ReadVariableOp&Adam/second/bias/m/Read/ReadVariableOp'Adam/third/kernel/m/Read/ReadVariableOp%Adam/third/bias/m/Read/ReadVariableOp(Adam/fourth/kernel/m/Read/ReadVariableOp&Adam/fourth/bias/m/Read/ReadVariableOp'Adam/first/kernel/v/Read/ReadVariableOp%Adam/first/bias/v/Read/ReadVariableOp(Adam/second/kernel/v/Read/ReadVariableOp&Adam/second/bias/v/Read/ReadVariableOp'Adam/third/kernel/v/Read/ReadVariableOp%Adam/third/bias/v/Read/ReadVariableOp(Adam/fourth/kernel/v/Read/ReadVariableOp&Adam/fourth/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1377048
½
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefirst/kernel
first/biassecond/kernelsecond/biasthird/kernel
third/biasfourth/kernelfourth/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/first/kernel/mAdam/first/bias/mAdam/second/kernel/mAdam/second/bias/mAdam/third/kernel/mAdam/third/bias/mAdam/fourth/kernel/mAdam/fourth/bias/mAdam/first/kernel/vAdam/first/bias/vAdam/second/kernel/vAdam/second/bias/vAdam/third/kernel/vAdam/third/bias/vAdam/fourth/kernel/vAdam/fourth/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1377151ûü


ô
C__inference_fourth_layer_call_and_return_conditional_losses_1376842

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ù	
Ã
/__inference_sequential_32_layer_call_fn_1376431
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376412o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input
£
ì
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376606
first_input
first_1376585:

first_1376587:
 
second_1376590:


second_1376592:

third_1376595:


third_1376597:
 
fourth_1376600:

fourth_1376602:
identity¢first/StatefulPartitionedCall¢fourth/StatefulPartitionedCall¢second/StatefulPartitionedCall¢third/StatefulPartitionedCallì
first/StatefulPartitionedCallStatefulPartitionedCallfirst_inputfirst_1376585first_1376587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_first_layer_call_and_return_conditional_losses_1376354
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_1376590second_1376592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_second_layer_call_and_return_conditional_losses_1376371
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_1376595third_1376597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_third_layer_call_and_return_conditional_losses_1376388
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_1376600fourth_1376602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_fourth_layer_call_and_return_conditional_losses_1376405v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input
ÁB
µ
 __inference__traced_save_1377048
file_prefix+
'savev2_first_kernel_read_readvariableop)
%savev2_first_bias_read_readvariableop,
(savev2_second_kernel_read_readvariableop*
&savev2_second_bias_read_readvariableop+
'savev2_third_kernel_read_readvariableop)
%savev2_third_bias_read_readvariableop,
(savev2_fourth_kernel_read_readvariableop*
&savev2_fourth_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_first_kernel_m_read_readvariableop0
,savev2_adam_first_bias_m_read_readvariableop3
/savev2_adam_second_kernel_m_read_readvariableop1
-savev2_adam_second_bias_m_read_readvariableop2
.savev2_adam_third_kernel_m_read_readvariableop0
,savev2_adam_third_bias_m_read_readvariableop3
/savev2_adam_fourth_kernel_m_read_readvariableop1
-savev2_adam_fourth_bias_m_read_readvariableop2
.savev2_adam_first_kernel_v_read_readvariableop0
,savev2_adam_first_bias_v_read_readvariableop3
/savev2_adam_second_kernel_v_read_readvariableop1
-savev2_adam_second_bias_v_read_readvariableop2
.savev2_adam_third_kernel_v_read_readvariableop0
,savev2_adam_third_bias_v_read_readvariableop3
/savev2_adam_fourth_kernel_v_read_readvariableop1
-savev2_adam_fourth_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ã
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì
valueâBß B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_first_kernel_read_readvariableop%savev2_first_bias_read_readvariableop(savev2_second_kernel_read_readvariableop&savev2_second_bias_read_readvariableop'savev2_third_kernel_read_readvariableop%savev2_third_bias_read_readvariableop(savev2_fourth_kernel_read_readvariableop&savev2_fourth_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_first_kernel_m_read_readvariableop,savev2_adam_first_bias_m_read_readvariableop/savev2_adam_second_kernel_m_read_readvariableop-savev2_adam_second_bias_m_read_readvariableop.savev2_adam_third_kernel_m_read_readvariableop,savev2_adam_third_bias_m_read_readvariableop/savev2_adam_fourth_kernel_m_read_readvariableop-savev2_adam_fourth_bias_m_read_readvariableop.savev2_adam_first_kernel_v_read_readvariableop,savev2_adam_first_bias_v_read_readvariableop/savev2_adam_second_kernel_v_read_readvariableop-savev2_adam_second_bias_v_read_readvariableop.savev2_adam_third_kernel_v_read_readvariableop,savev2_adam_third_bias_v_read_readvariableop/savev2_adam_fourth_kernel_v_read_readvariableop-savev2_adam_fourth_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ç
_input_shapesÕ
Ò: :
:
:

:
:

:
:
:: : : : : : : :
:
:

:
:

:
:
::
:
:

:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
:: 

_output_shapes
: 
´'

J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755

inputs6
$first_matmul_readvariableop_resource:
3
%first_biasadd_readvariableop_resource:
7
%second_matmul_readvariableop_resource:

4
&second_biasadd_readvariableop_resource:
6
$third_matmul_readvariableop_resource:

3
%third_biasadd_readvariableop_resource:
7
%fourth_matmul_readvariableop_resource:
4
&fourth_biasadd_readvariableop_resource:
identity¢first/BiasAdd/ReadVariableOp¢first/MatMul/ReadVariableOp¢fourth/BiasAdd/ReadVariableOp¢fourth/MatMul/ReadVariableOp¢second/BiasAdd/ReadVariableOp¢second/MatMul/ReadVariableOp¢third/BiasAdd/ReadVariableOp¢third/MatMul/ReadVariableOp
first/MatMul/ReadVariableOpReadVariableOp$first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0u
first/MatMulMatMulinputs#first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
first/BiasAdd/ReadVariableOpReadVariableOp%first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
first/BiasAddBiasAddfirst/MatMul:product:0$first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
O

first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?o
	first/mulMulfirst/beta:output:0first/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
first/SigmoidSigmoidfirst/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
first/mul_1Mulfirst/BiasAdd:output:0first/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
first/IdentityIdentityfirst/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
first/IdentityN	IdentityNfirst/mul_1:z:0first/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-1376726*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

second/MatMul/ReadVariableOpReadVariableOp%second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
second/MatMulMatMulfirst/IdentityN:output:0$second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

second/BiasAdd/ReadVariableOpReadVariableOp&second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
second/BiasAddBiasAddsecond/MatMul:product:0%second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
second/SigmoidSigmoidsecond/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

third/MatMul/ReadVariableOpReadVariableOp$third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
third/MatMulMatMulsecond/Sigmoid:y:0#third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
third/BiasAdd/ReadVariableOpReadVariableOp%third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
third/BiasAddBiasAddthird/MatMul:product:0$third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
third/SigmoidSigmoidthird/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

fourth/MatMul/ReadVariableOpReadVariableOp%fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
fourth/MatMulMatMulthird/Sigmoid:y:0$fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fourth/BiasAdd/ReadVariableOpReadVariableOp&fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fourth/BiasAddBiasAddfourth/MatMul:product:0%fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
fourth/SeluSelufourth/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityfourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp^first/BiasAdd/ReadVariableOp^first/MatMul/ReadVariableOp^fourth/BiasAdd/ReadVariableOp^fourth/MatMul/ReadVariableOp^second/BiasAdd/ReadVariableOp^second/MatMul/ReadVariableOp^third/BiasAdd/ReadVariableOp^third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
first/BiasAdd/ReadVariableOpfirst/BiasAdd/ReadVariableOp2:
first/MatMul/ReadVariableOpfirst/MatMul/ReadVariableOp2>
fourth/BiasAdd/ReadVariableOpfourth/BiasAdd/ReadVariableOp2<
fourth/MatMul/ReadVariableOpfourth/MatMul/ReadVariableOp2>
second/BiasAdd/ReadVariableOpsecond/BiasAdd/ReadVariableOp2<
second/MatMul/ReadVariableOpsecond/MatMul/ReadVariableOp2<
third/BiasAdd/ReadVariableOpthird/BiasAdd/ReadVariableOp2:
third/MatMul/ReadVariableOpthird/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
¾
/__inference_sequential_32_layer_call_fn_1376656

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376412o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

'__inference_third_layer_call_fn_1376811

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_third_layer_call_and_return_conditional_losses_1376388o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
´'

J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716

inputs6
$first_matmul_readvariableop_resource:
3
%first_biasadd_readvariableop_resource:
7
%second_matmul_readvariableop_resource:

4
&second_biasadd_readvariableop_resource:
6
$third_matmul_readvariableop_resource:

3
%third_biasadd_readvariableop_resource:
7
%fourth_matmul_readvariableop_resource:
4
&fourth_biasadd_readvariableop_resource:
identity¢first/BiasAdd/ReadVariableOp¢first/MatMul/ReadVariableOp¢fourth/BiasAdd/ReadVariableOp¢fourth/MatMul/ReadVariableOp¢second/BiasAdd/ReadVariableOp¢second/MatMul/ReadVariableOp¢third/BiasAdd/ReadVariableOp¢third/MatMul/ReadVariableOp
first/MatMul/ReadVariableOpReadVariableOp$first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0u
first/MatMulMatMulinputs#first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
first/BiasAdd/ReadVariableOpReadVariableOp%first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
first/BiasAddBiasAddfirst/MatMul:product:0$first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
O

first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?o
	first/mulMulfirst/beta:output:0first/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
first/SigmoidSigmoidfirst/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
first/mul_1Mulfirst/BiasAdd:output:0first/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
first/IdentityIdentityfirst/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
first/IdentityN	IdentityNfirst/mul_1:z:0first/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-1376687*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

second/MatMul/ReadVariableOpReadVariableOp%second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
second/MatMulMatMulfirst/IdentityN:output:0$second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

second/BiasAdd/ReadVariableOpReadVariableOp&second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
second/BiasAddBiasAddsecond/MatMul:product:0%second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
second/SigmoidSigmoidsecond/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

third/MatMul/ReadVariableOpReadVariableOp$third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
third/MatMulMatMulsecond/Sigmoid:y:0#third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
third/BiasAdd/ReadVariableOpReadVariableOp%third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
third/BiasAddBiasAddthird/MatMul:product:0$third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
third/SigmoidSigmoidthird/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

fourth/MatMul/ReadVariableOpReadVariableOp%fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
fourth/MatMulMatMulthird/Sigmoid:y:0$fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fourth/BiasAdd/ReadVariableOpReadVariableOp&fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fourth/BiasAddBiasAddfourth/MatMul:product:0%fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
fourth/SeluSelufourth/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityfourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp^first/BiasAdd/ReadVariableOp^first/MatMul/ReadVariableOp^fourth/BiasAdd/ReadVariableOp^fourth/MatMul/ReadVariableOp^second/BiasAdd/ReadVariableOp^second/MatMul/ReadVariableOp^third/BiasAdd/ReadVariableOp^third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
first/BiasAdd/ReadVariableOpfirst/BiasAdd/ReadVariableOp2:
first/MatMul/ReadVariableOpfirst/MatMul/ReadVariableOp2>
fourth/BiasAdd/ReadVariableOpfourth/BiasAdd/ReadVariableOp2<
fourth/MatMul/ReadVariableOpfourth/MatMul/ReadVariableOp2>
second/BiasAdd/ReadVariableOpsecond/BiasAdd/ReadVariableOp2<
second/MatMul/ReadVariableOpsecond/MatMul/ReadVariableOp2<
third/BiasAdd/ReadVariableOpthird/BiasAdd/ReadVariableOp2:
third/MatMul/ReadVariableOpthird/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

$__inference_internal_grad_fn_1376962
result_grads_0
result_grads_1
mul_first_beta
mul_first_biasadd
identityp
mulMulmul_first_betamul_first_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
mul_1Mulmul_first_betamul_first_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
: :ÿÿÿÿÿÿÿÿÿ
:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

£
ì
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376582
first_input
first_1376561:

first_1376563:
 
second_1376566:


second_1376568:

third_1376571:


third_1376573:
 
fourth_1376576:

fourth_1376578:
identity¢first/StatefulPartitionedCall¢fourth/StatefulPartitionedCall¢second/StatefulPartitionedCall¢third/StatefulPartitionedCallì
first/StatefulPartitionedCallStatefulPartitionedCallfirst_inputfirst_1376561first_1376563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_first_layer_call_and_return_conditional_losses_1376354
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_1376566second_1376568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_second_layer_call_and_return_conditional_losses_1376371
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_1376571third_1376573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_third_layer_call_and_return_conditional_losses_1376388
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_1376576fourth_1376578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_fourth_layer_call_and_return_conditional_losses_1376405v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input
Â

$__inference_internal_grad_fn_1376980
result_grads_0
result_grads_1
mul_first_beta
mul_first_biasadd
identityp
mulMulmul_first_betamul_first_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
mul_1Mulmul_first_betamul_first_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
: :ÿÿÿÿÿÿÿÿÿ
:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ



ó
B__inference_third_layer_call_and_return_conditional_losses_1376822

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ù	
Ã
/__inference_sequential_32_layer_call_fn_1376558
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376518o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input

£
$__inference_internal_grad_fn_1376926
result_grads_0
result_grads_1 
mul_sequential_32_first_beta#
mul_sequential_32_first_biasadd
identity
mulMulmul_sequential_32_first_betamul_sequential_32_first_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
mul_1Mulmul_sequential_32_first_betamul_sequential_32_first_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
: :ÿÿÿÿÿÿÿÿÿ
:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ



ô
C__inference_fourth_layer_call_and_return_conditional_losses_1376405

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitySelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¨
õ
B__inference_first_layer_call_and_return_conditional_losses_1376354

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:


identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-1376346*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
¾
/__inference_sequential_32_layer_call_fn_1376677

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376518o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ô
C__inference_second_layer_call_and_return_conditional_losses_1376802

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¾

'__inference_first_layer_call_fn_1376764

inputs
unknown:

	unknown_0:

identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_first_layer_call_and_return_conditional_losses_1376354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§	
¹
%__inference_signature_wrapper_1376635
first_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:


	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1376329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input

{
$__inference_internal_grad_fn_1376998
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
: :ÿÿÿÿÿÿÿÿÿ
:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

¨
õ
B__inference_first_layer_call_and_return_conditional_losses_1376782

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:


identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-1376774*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

(__inference_fourth_layer_call_fn_1376831

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_fourth_layer_call_and_return_conditional_losses_1376405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

ç
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376412

inputs
first_1376355:

first_1376357:
 
second_1376372:


second_1376374:

third_1376389:


third_1376391:
 
fourth_1376406:

fourth_1376408:
identity¢first/StatefulPartitionedCall¢fourth/StatefulPartitionedCall¢second/StatefulPartitionedCall¢third/StatefulPartitionedCallç
first/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_1376355first_1376357*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_first_layer_call_and_return_conditional_losses_1376354
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_1376372second_1376374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_second_layer_call_and_return_conditional_losses_1376371
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_1376389third_1376391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_third_layer_call_and_return_conditional_losses_1376388
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_1376406fourth_1376408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_fourth_layer_call_and_return_conditional_losses_1376405v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

(__inference_second_layer_call_fn_1376791

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_second_layer_call_and_return_conditional_losses_1376371o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
|

#__inference__traced_restore_1377151
file_prefix/
assignvariableop_first_kernel:
+
assignvariableop_1_first_bias:
2
 assignvariableop_2_second_kernel:

,
assignvariableop_3_second_bias:
1
assignvariableop_4_third_kernel:

+
assignvariableop_5_third_bias:
2
 assignvariableop_6_fourth_kernel:
,
assignvariableop_7_fourth_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: 9
'assignvariableop_15_adam_first_kernel_m:
3
%assignvariableop_16_adam_first_bias_m:
:
(assignvariableop_17_adam_second_kernel_m:

4
&assignvariableop_18_adam_second_bias_m:
9
'assignvariableop_19_adam_third_kernel_m:

3
%assignvariableop_20_adam_third_bias_m:
:
(assignvariableop_21_adam_fourth_kernel_m:
4
&assignvariableop_22_adam_fourth_bias_m:9
'assignvariableop_23_adam_first_kernel_v:
3
%assignvariableop_24_adam_first_bias_v:
:
(assignvariableop_25_adam_second_kernel_v:

4
&assignvariableop_26_adam_second_bias_v:
9
'assignvariableop_27_adam_third_kernel_v:

3
%assignvariableop_28_adam_third_bias_v:
:
(assignvariableop_29_adam_fourth_kernel_v:
4
&assignvariableop_30_adam_fourth_bias_v:
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Æ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì
valueâBß B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_first_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_first_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_second_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_second_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_third_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_third_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_fourth_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_fourth_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_first_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_first_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_second_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_second_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_third_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_third_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_fourth_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_fourth_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_first_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_first_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_second_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_second_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_third_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_third_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_fourth_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_fourth_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ô
C__inference_second_layer_call_and_return_conditional_losses_1376371

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
2
Ö
"__inference__wrapped_model_1376329
first_inputD
2sequential_32_first_matmul_readvariableop_resource:
A
3sequential_32_first_biasadd_readvariableop_resource:
E
3sequential_32_second_matmul_readvariableop_resource:

B
4sequential_32_second_biasadd_readvariableop_resource:
D
2sequential_32_third_matmul_readvariableop_resource:

A
3sequential_32_third_biasadd_readvariableop_resource:
E
3sequential_32_fourth_matmul_readvariableop_resource:
B
4sequential_32_fourth_biasadd_readvariableop_resource:
identity¢*sequential_32/first/BiasAdd/ReadVariableOp¢)sequential_32/first/MatMul/ReadVariableOp¢+sequential_32/fourth/BiasAdd/ReadVariableOp¢*sequential_32/fourth/MatMul/ReadVariableOp¢+sequential_32/second/BiasAdd/ReadVariableOp¢*sequential_32/second/MatMul/ReadVariableOp¢*sequential_32/third/BiasAdd/ReadVariableOp¢)sequential_32/third/MatMul/ReadVariableOp
)sequential_32/first/MatMul/ReadVariableOpReadVariableOp2sequential_32_first_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
sequential_32/first/MatMulMatMulfirst_input1sequential_32/first/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*sequential_32/first/BiasAdd/ReadVariableOpReadVariableOp3sequential_32_first_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0²
sequential_32/first/BiasAddBiasAdd$sequential_32/first/MatMul:product:02sequential_32/first/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
sequential_32/first/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
sequential_32/first/mulMul!sequential_32/first/beta:output:0$sequential_32/first/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
sequential_32/first/SigmoidSigmoidsequential_32/first/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential_32/first/mul_1Mul$sequential_32/first/BiasAdd:output:0sequential_32/first/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
sequential_32/first/IdentityIdentitysequential_32/first/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ç
sequential_32/first/IdentityN	IdentityNsequential_32/first/mul_1:z:0$sequential_32/first/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-1376300*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

*sequential_32/second/MatMul/ReadVariableOpReadVariableOp3sequential_32_second_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0³
sequential_32/second/MatMulMatMul&sequential_32/first/IdentityN:output:02sequential_32/second/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

+sequential_32/second/BiasAdd/ReadVariableOpReadVariableOp4sequential_32_second_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0µ
sequential_32/second/BiasAddBiasAdd%sequential_32/second/MatMul:product:03sequential_32/second/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential_32/second/SigmoidSigmoid%sequential_32/second/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)sequential_32/third/MatMul/ReadVariableOpReadVariableOp2sequential_32_third_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0«
sequential_32/third/MatMulMatMul sequential_32/second/Sigmoid:y:01sequential_32/third/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*sequential_32/third/BiasAdd/ReadVariableOpReadVariableOp3sequential_32_third_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0²
sequential_32/third/BiasAddBiasAdd$sequential_32/third/MatMul:product:02sequential_32/third/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
sequential_32/third/SigmoidSigmoid$sequential_32/third/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*sequential_32/fourth/MatMul/ReadVariableOpReadVariableOp3sequential_32_fourth_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¬
sequential_32/fourth/MatMulMatMulsequential_32/third/Sigmoid:y:02sequential_32/fourth/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_32/fourth/BiasAdd/ReadVariableOpReadVariableOp4sequential_32_fourth_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_32/fourth/BiasAddBiasAdd%sequential_32/fourth/MatMul:product:03sequential_32/fourth/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_32/fourth/SeluSelu%sequential_32/fourth/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_32/fourth/Selu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp+^sequential_32/first/BiasAdd/ReadVariableOp*^sequential_32/first/MatMul/ReadVariableOp,^sequential_32/fourth/BiasAdd/ReadVariableOp+^sequential_32/fourth/MatMul/ReadVariableOp,^sequential_32/second/BiasAdd/ReadVariableOp+^sequential_32/second/MatMul/ReadVariableOp+^sequential_32/third/BiasAdd/ReadVariableOp*^sequential_32/third/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2X
*sequential_32/first/BiasAdd/ReadVariableOp*sequential_32/first/BiasAdd/ReadVariableOp2V
)sequential_32/first/MatMul/ReadVariableOp)sequential_32/first/MatMul/ReadVariableOp2Z
+sequential_32/fourth/BiasAdd/ReadVariableOp+sequential_32/fourth/BiasAdd/ReadVariableOp2X
*sequential_32/fourth/MatMul/ReadVariableOp*sequential_32/fourth/MatMul/ReadVariableOp2Z
+sequential_32/second/BiasAdd/ReadVariableOp+sequential_32/second/BiasAdd/ReadVariableOp2X
*sequential_32/second/MatMul/ReadVariableOp*sequential_32/second/MatMul/ReadVariableOp2X
*sequential_32/third/BiasAdd/ReadVariableOp*sequential_32/third/BiasAdd/ReadVariableOp2V
)sequential_32/third/MatMul/ReadVariableOp)sequential_32/third/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namefirst_input

ç
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376518

inputs
first_1376497:

first_1376499:
 
second_1376502:


second_1376504:

third_1376507:


third_1376509:
 
fourth_1376512:

fourth_1376514:
identity¢first/StatefulPartitionedCall¢fourth/StatefulPartitionedCall¢second/StatefulPartitionedCall¢third/StatefulPartitionedCallç
first/StatefulPartitionedCallStatefulPartitionedCallinputsfirst_1376497first_1376499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_first_layer_call_and_return_conditional_losses_1376354
second/StatefulPartitionedCallStatefulPartitionedCall&first/StatefulPartitionedCall:output:0second_1376502second_1376504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_second_layer_call_and_return_conditional_losses_1376371
third/StatefulPartitionedCallStatefulPartitionedCall'second/StatefulPartitionedCall:output:0third_1376507third_1376509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_third_layer_call_and_return_conditional_losses_1376388
fourth/StatefulPartitionedCallStatefulPartitionedCall&third/StatefulPartitionedCall:output:0fourth_1376512fourth_1376514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_fourth_layer_call_and_return_conditional_losses_1376405v
IdentityIdentity'fourth/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^first/StatefulPartitionedCall^fourth/StatefulPartitionedCall^second/StatefulPartitionedCall^third/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
first/StatefulPartitionedCallfirst/StatefulPartitionedCall2@
fourth/StatefulPartitionedCallfourth/StatefulPartitionedCall2@
second/StatefulPartitionedCallsecond/StatefulPartitionedCall2>
third/StatefulPartitionedCallthird/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_third_layer_call_and_return_conditional_losses_1376388

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

{
$__inference_internal_grad_fn_1376944
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
: :ÿÿÿÿÿÿÿÿÿ
:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>
$__inference_internal_grad_fn_1376926CustomGradient-1376300>
$__inference_internal_grad_fn_1376944CustomGradient-1376346>
$__inference_internal_grad_fn_1376962CustomGradient-1376687>
$__inference_internal_grad_fn_1376980CustomGradient-1376726>
$__inference_internal_grad_fn_1376998CustomGradient-1376774"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
C
first_input4
serving_default_first_input:0ÿÿÿÿÿÿÿÿÿ:
fourth0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:þ

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
»
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
X
0
1
2
3
%4
&5
-6
.7"
trackable_list_wrapper
X
0
1
2
3
%4
&5
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ò
4trace_0
5trace_1
6trace_2
7trace_32
/__inference_sequential_32_layer_call_fn_1376431
/__inference_sequential_32_layer_call_fn_1376656
/__inference_sequential_32_layer_call_fn_1376677
/__inference_sequential_32_layer_call_fn_1376558À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z4trace_0z5trace_1z6trace_2z7trace_3
Þ
8trace_0
9trace_1
:trace_2
;trace_32ó
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376582
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376606À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z8trace_0z9trace_1z:trace_2z;trace_3
ÑBÎ
"__inference__wrapped_model_1376329first_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ã
<iter

=beta_1

>beta_2
	?decay
@learning_ratemcmdmemf%mg&mh-mi.mjvkvlvmvn%vo&vp-vq.vr"
	optimizer
¨2¥¢
²
FullArgSpec
args
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Aserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
Gtrace_02Î
'__inference_first_layer_call_fn_1376764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zGtrace_0

Htrace_02é
B__inference_first_layer_call_and_return_conditional_losses_1376782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zHtrace_0
:
2first/kernel
:
2
first/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ì
Ntrace_02Ï
(__inference_second_layer_call_fn_1376791¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zNtrace_0

Otrace_02ê
C__inference_second_layer_call_and_return_conditional_losses_1376802¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zOtrace_0
:

2second/kernel
:
2second/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ë
Utrace_02Î
'__inference_third_layer_call_fn_1376811¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zUtrace_0

Vtrace_02é
B__inference_third_layer_call_and_return_conditional_losses_1376822¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zVtrace_0
:

2third/kernel
:
2
third/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ì
\trace_02Ï
(__inference_fourth_layer_call_fn_1376831¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z\trace_0

]trace_02ê
C__inference_fourth_layer_call_and_return_conditional_losses_1376842¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z]trace_0
:
2fourth/kernel
:2fourth/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_32_layer_call_fn_1376431first_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
/__inference_sequential_32_layer_call_fn_1376656inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
/__inference_sequential_32_layer_call_fn_1376677inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
/__inference_sequential_32_layer_call_fn_1376558first_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¡B
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376582first_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¡B
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376606first_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÐBÍ
%__inference_signature_wrapper_1376635first_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_first_layer_call_fn_1376764inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_first_layer_call_and_return_conditional_losses_1376782inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_second_layer_call_fn_1376791inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_second_layer_call_and_return_conditional_losses_1376802inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_third_layer_call_fn_1376811inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_third_layer_call_and_return_conditional_losses_1376822inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_fourth_layer_call_fn_1376831inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_fourth_layer_call_and_return_conditional_losses_1376842inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
_	variables
`	keras_api
	atotal
	bcount"
_tf_keras_metric
.
a0
b1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
#:!
2Adam/first/kernel/m
:
2Adam/first/bias/m
$:"

2Adam/second/kernel/m
:
2Adam/second/bias/m
#:!

2Adam/third/kernel/m
:
2Adam/third/bias/m
$:"
2Adam/fourth/kernel/m
:2Adam/fourth/bias/m
#:!
2Adam/first/kernel/v
:
2Adam/first/bias/v
$:"

2Adam/second/kernel/v
:
2Adam/second/bias/v
#:!

2Adam/third/kernel/v
:
2Adam/third/bias/v
$:"
2Adam/fourth/kernel/v
:2Adam/fourth/bias/v
Bb@
sequential_32/first/beta:0"__inference__wrapped_model_1376329
EbC
sequential_32/first/BiasAdd:0"__inference__wrapped_model_1376329
NbL
beta:0B__inference_first_layer_call_and_return_conditional_losses_1376354
QbO
	BiasAdd:0B__inference_first_layer_call_and_return_conditional_losses_1376354
\bZ
first/beta:0J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716
_b]
first/BiasAdd:0J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716
\bZ
first/beta:0J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755
_b]
first/BiasAdd:0J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755
NbL
beta:0B__inference_first_layer_call_and_return_conditional_losses_1376782
QbO
	BiasAdd:0B__inference_first_layer_call_and_return_conditional_losses_1376782
"__inference__wrapped_model_1376329q%&-.4¢1
*¢'
%"
first_inputÿÿÿÿÿÿÿÿÿ
ª "/ª,
*
fourth 
fourthÿÿÿÿÿÿÿÿÿ¢
B__inference_first_layer_call_and_return_conditional_losses_1376782\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_first_layer_call_fn_1376764O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
£
C__inference_fourth_layer_call_and_return_conditional_losses_1376842\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_fourth_layer_call_fn_1376831O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿº
$__inference_internal_grad_fn_1376926ste¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ

(%
result_grads_1ÿÿÿÿÿÿÿÿÿ

ª "$!

 

1ÿÿÿÿÿÿÿÿÿ
º
$__inference_internal_grad_fn_1376944uve¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ

(%
result_grads_1ÿÿÿÿÿÿÿÿÿ

ª "$!

 

1ÿÿÿÿÿÿÿÿÿ
º
$__inference_internal_grad_fn_1376962wxe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ

(%
result_grads_1ÿÿÿÿÿÿÿÿÿ

ª "$!

 

1ÿÿÿÿÿÿÿÿÿ
º
$__inference_internal_grad_fn_1376980yze¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ

(%
result_grads_1ÿÿÿÿÿÿÿÿÿ

ª "$!

 

1ÿÿÿÿÿÿÿÿÿ
º
$__inference_internal_grad_fn_1376998{|e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ

(%
result_grads_1ÿÿÿÿÿÿÿÿÿ

ª "$!

 

1ÿÿÿÿÿÿÿÿÿ
£
C__inference_second_layer_call_and_return_conditional_losses_1376802\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 {
(__inference_second_layer_call_fn_1376791O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
½
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376582o%&-.<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376606o%&-.<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376716j%&-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_sequential_32_layer_call_and_return_conditional_losses_1376755j%&-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_32_layer_call_fn_1376431b%&-.<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_32_layer_call_fn_1376558b%&-.<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_32_layer_call_fn_1376656]%&-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_32_layer_call_fn_1376677]%&-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿª
%__inference_signature_wrapper_1376635%&-.C¢@
¢ 
9ª6
4
first_input%"
first_inputÿÿÿÿÿÿÿÿÿ"/ª,
*
fourth 
fourthÿÿÿÿÿÿÿÿÿ¢
B__inference_third_layer_call_and_return_conditional_losses_1376822\%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_third_layer_call_fn_1376811O%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
