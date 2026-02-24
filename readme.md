MetaVision FoundationPoseLive Client 
---

Dieser Client dient als Benutzeroberfläche für das MetaVision Tracking-System. Eine Verbindung wird zu einem dedizierten Tracking-Server hergestellt, wobei Bilddaten über eine Intel RealSense Kamera erfasst und die vom Server berechneten Posen in Echtzeit auf dem Live-Videostream visualisiert werden.

### Voraussetzungen

Es muss sichergestellt sein, dass eine Intel RealSense Kamera per USB3.0 angeschlossen ist. Zudem ist zu gewährleisten, dass der MetaVision Server im Netzwerk erreichbar ist. Die Installation der erforderlichen Python-Pakete kann über den folgenden Befehl automatisiert vorgenommen werden:

    `pip install -r requirements.txt`

Es wird empfohlen die Abhängigkeit in einer virtuellen Umgebung zu installieren.

Nutzung
---
### Programmstart und Serververbindung

Die Anwendung kann durch Ausführen des mitgelieferten Executables gestartet werden, alternativ ist auch ein manuelles Ausführen des MTFPL_client_gui.py Skripts möglich. Nach dem Start erscheint ein Dialog zur Serververbindung. Hier ist die IP-Adresse des Rechners einzutragen, auf dem der Tracking-Server betrieben wird. Durch Betätigung von Verbindung prüfen wird der Status abgefragt. Bei erfolgreicher Rückmeldung des Servers öffnet sich das Hauptfenster.

### Setup (Schritt 1 bis 3)
Vor dem Start des Trackings ist das Setup in der Seitenleiste vollständig abzuschließen. Der Start-Button wird erst freigeschaltet, sobald alle notwendigen Informationen hinterlegt wurden.

Schritt 1: Upload CAD Model. Die zu trackende OBJ-Datei ist vom lokalen PC auszuwählen. Das Modell wird zum Server übertragen und in der 3D-Vorschau dargestellt.

Schritt 2: Definition des Aussehens. Es kann entweder über die Farbauswahl eine Volltonfarbe bestimmt oder über die Textur-Funktion eine Oberfläche vom Server geladen werden. Die Auswahl erfolgt per Doppelklick aus der bereitgestellten Liste.

Schritt 3: Erstellung der Maske. Nach Betätigung des Buttons sind durch zwei Linksklicks im Live-Kamerabild die Eckpunkte eines Rechtecks um das zu trackende Objekt festzulegen.

### Tracking starten und stoppen
Durch Klicken auf Start Tracking wird der Vorgang eingeleitet. Die Datenübertragung zum Server beginnt, und die Visualisierung der Bounding-Box sowie der Koordinatenachsen erfolgt im Video-Feed. Der Vorgang kann über Stop Tracking beendet werden.

### Speicherung der Logs
Nach Beendigung des Trackings können die aufgezeichneten Daten der Sitzung über die Log-Download-Funktion als LOG-Datei gesichert werden. Diese enthält Bild-IDs, Zeitstempel und die entsprechenden Pose-Matrizen.