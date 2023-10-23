'''
  For more samples please visit https://github.com/Azure-Samples/cognitive-services-speech-sdk
'''

import azure.cognitiveservices.speech as speechsdk


import os
text = "Hola, soy Maria Jose"
# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription='fae9922ffa8146ae82ca74722dcbf955',
                                       region='eastus')
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) #Reproduce
#audio_config = speechsdk.audio.AudioOutputConfig(filename="/home/maria/pepper/file.wav")
# Set either the `SpeechSynthesisVoiceName` or `SpeechSynthesisLanguage`.
speech_config.speech_synthesis_language = "es-CO"
speech_config.speech_synthesis_voice_name ="es-CO-GonzaloNeural"

#
# speech_config.speech_synthesis_voice_style = 'Friendly'
# speech_config.speech_synthesis_voice_role= 'Girl'

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that speaks.
# speech_synthesis_voice_name = 'es-CO-GonzaloNeural'  # nl-BE-DenaNeural Female # nl-BE-ArnaudNeural (Male)
ssml = """<speak version='1.0'  xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-CO'>
    <voice name='es-CO-GonzaloNeural'>
     <mstts:express-as role="Boy" style ="calm">
        hola me llamo Maria
     </mstts:express-as>
    </voice>
</speak>"""


# Synthesize the SSML
print("SSML to synthesize: \r\n{}".format(ssml))
speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()
#
# speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
print(type(speech_synthesis_result))


# def agent_voice(text):
#     # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
#     speech_config = speechsdk.SpeechConfig(subscription='fae9922ffa8146ae82ca74722dcbf955',
#                                            region='eastus')
#     audio_config = speechsdk.audio.AudioOutputConfig(filename="/home/maria/pepper/file.wav")
#     # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) #Reproduce
#
#     # The language of the voice that speaks.
#     speech_config.speech_synthesis_voice_name = 'nl-BE-ArnaudNeural'  # nl-BE-DenaNeural Female # nl-BE-ArnaudNeural (Male)
#     speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
#     speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
#
#     if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#         print("Speech synthesized for text [{}]".format(text))
#     elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
#         cancellation_details = speech_synthesis_result.cancellation_details
#         print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#         if cancellation_details.reason == speechsdk.CancellationReason.Error:
#             if cancellation_details.error_details:
#                 print("Error details: {}".format(cancellation_details.error_details))
#                 print("Did you set the speech resource key and region values?")
#
# def main():
#      program()


# if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#     print("Speech synthesized for text [{}]".format(text))
# elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
#     cancellation_details = speech_synthesis_result.cancellation_details
#     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#     if cancellation_details.reason == speechsdk.CancellationReason.Error:
#         if cancellation_details.error_details:
#             print("Error details: {}".format(cancellation_details.error_details))
#             print("Did you set the speech resource key and region values?")

#
# speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# # Note: the voice setting will not overwrite the voice element in input SSML.
# speech_config.speech_synthesis_voice_name = "es-CO-SalomeNeural"
#
# text = "Hi, this is Salome"
#
# # use the default speaker as audio output.
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
#
# result = speech_synthesizer.speak_text_async(text).get()
# # Check result
# if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#     print("Speech synthesized for text [{}]".format(text))
# elif result.reason == speechsdk.ResultReason.Canceled:
#     cancellation_details = result.cancellation_details
#     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#     if cancellation_details.reason == speechsdk.CancellationReason.Error:
#         print("Error details: {}".format(cancellation_details.error_details))

#
