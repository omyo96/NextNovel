import React, { useEffect, useRef, useState, useReducer } from "react";
import style from "./Canvas1.module.css";

export default function Canvas1({ imageSrcs, setImageSrcs, selected }) {
  const canvasRef = useRef(null); //canvas
  const [getCtx, setGetCtx] = useState(null); //canvas
  const [painting, setPainting] = useState(false); //그림을 그리고 있는지 아닌지
  const [mouseX, setmouseX] = useState(); //캔버스 내 마우스 좌표
  const [mouseY, setmouseY] = useState(); //캔버스 내 마우스 좌표
  const canvasWidth = 608;
  const canvasHeight = 380;

  const [widthState, setWidthState] = useState(2.5); //펜 굵기 초기값
  const [colorState, setColorState] = useState("#000000"); //펜 색 초기값
  const [openSetWidthState, setOpenSetWidthState] = useState(false); //펜 굵기 설정 탭 on/off
  const [openSetColorState, setOpenSetColorState] = useState(false); //펜 색 설정 탭 on/off
  const [store, dispatch] = useReducer(reducer, [imageSrcs[selected]]); //뒤로가기 저장소
  const [paintState, setPaintState] = useState(false); //캔버스 내 마우스 클릭중 or 클릭해제, 벗어남
  const colors = [
    "#e53730",
    "#d81758",
    "#8a23a4",
    "#5a34ad",
    "#3c49ab",
    "#3e8cef",
    "#3fa0f1",
    "#44b4cd",
    "#328b7d",
    "#55a549",
    "#87bb44",
    "#c7d737",
    "#fce739",
    "#f7b816",
    "#f48c10",
    "#f14b20",
    "#6a4b3f",
    "#597280",
    "#c1c1c1",
    "#6f6f6f",
    "#000000",
  ]; //컬러파레트 색상

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    const ctx = canvas.getContext("2d");
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = widthState;
    ctx.strokeStyle = colorState;
    setGetCtx(ctx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); //맨 처음 컴포넌트 설정

  useEffect(() => {
    if (getCtx) getCtx.clearRect(0, 0, canvasWidth, canvasHeight); //현재 캔버스 초기화

    const img = new Image();
    img.src = imageSrcs[selected];
    img.onload = () => getCtx.drawImage(img, 0, 0); //캔버스 불러오기

    dispatch({ type: "init" }); //저장소 초기화
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]); //n번째 캔버스 선택시

  useEffect(() => {
    if (!painting && paintState) {
      //그림그리는상태가 아니고, 마우스에서 손이 떨어졌을 때
      const canvas = canvasRef.current;
      const dataURL = canvas.toDataURL();
      setImageSrcs(
        imageSrcs.map((imageSrc, index) =>
          index === selected ? dataURL : imageSrc
        )
      ); //현재 캔버스를 완성그림에 저장하고
      dispatch({ type: "increment", dataURL }); //저장소에 기록을 추가
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [painting]); //그림그리는 행위를 하는 상태

  function reducer(state, action) {
    //저장소 간리
    switch (action.type) {
      case "increment": //그리기
        return [...state, action.dataURL];
      case "decrement": //뒤로가기
        return [...state.slice(0, state.length - 1)];
      case "init": //초기화
        return [imageSrcs[selected]];
      default:
        throw new Error();
    }
  }

  const drawFn = (e) => {
    //마우스가 캔버스 위에 올라가있을때
    setmouseX(e.nativeEvent.offsetX);
    setmouseY(e.nativeEvent.offsetY);

    if (!painting) {
      //그리는 행위 중이 아님
      getCtx.beginPath();
      getCtx.moveTo(mouseX, mouseY);
    } else {
      //그리는 행위 중
      getCtx.lineTo(mouseX, mouseY);
      getCtx.stroke();
      setPaintState(true); //마우스에 손이 눌러지고 있음
    }
  };
  const onPencil = () => {
    //펜 선택
    getCtx.strokeStyle = colorState;
  };
  const onEraser = () => {
    //지우개 선택
    getCtx.strokeStyle = "white";
  };
  const openSetWidth = () => {
    //펜 굵기 설정 탭 열기
    setOpenSetWidthState((prev) => !prev);
  };
  const setWidth = (event) => {
    //펜 굵기 설정하기
    setWidthState(event.target.value);
    getCtx.lineWidth = event.target.value;
  };
  const openSetColor = () => {
    //펜 색 설정 탭 열기
    setOpenSetColorState((prev) => !prev);
  };
  const setColor = (event) => {
    //펜 색 설정하기
    setColorState(event.target.dataset.color);
    getCtx.strokeStyle = event.target.dataset.color;
  };
  const goBack = () => {
    //뒤로가기
    if (store.length === 1) return; //처음 상태면 return

    dispatch({ type: "decrement" }); //저장소에서 맨 뒤 지우기

    const dataURL = store[store.length - 2]; //dispatch가 비동기라서 -2를 하여 불러옴

    getCtx.clearRect(0, 0, canvasWidth, canvasHeight); //현재 캔버스 초기화

    const img = new Image();
    img.src = dataURL;
    img.onload = () => getCtx.drawImage(img, 0, 0); //이전 이미지 불러오기

    setImageSrcs(
      imageSrcs.map((imageSrc, index) =>
        index === selected ? dataURL : imageSrc
      )
    ); //완성 그림에 전달
  };
  const initCanvas = () => {
    //쓰레기통으로 캔버스 초기화
    getCtx.clearRect(0, 0, canvasWidth, canvasHeight); //현재 캔버스 초기화
    setImageSrcs(
      imageSrcs.map((imageSrc, index) =>
        index === selected ? undefined : imageSrc
      )
    ); //완성 그림에 undefined 전달
    dispatch({ type: "init" }); //저장소 초기화
  };

  return (
    <div className={style.container}>
      <div className={style.canvas}>
        <canvas
          ref={canvasRef}
          onMouseDown={() => setPainting(true)}
          onMouseUp={() => {
            setPainting(false);
          }}
          onMouseMove={(e) => drawFn(e)}
          onMouseLeave={() => {
            setPainting(false);
          }}
        ></canvas>
      </div>
      <div className={style.tools}>
        <div className={style.tool1}>
          <img
            src={process.env.PUBLIC_URL + `/icon/pen.svg`}
            alt="pen"
            onClick={onPencil}
          />
        </div>
        <div className={style.tool2}>
          <img
            src={process.env.PUBLIC_URL + `/icon/eraser.svg`}
            alt="eraser"
            onClick={onEraser}
          />
        </div>
        <div className={style.tool3}>
          <div
            style={{
              width: "50px",
              height: "30px",
              backgroundColor: `${colorState}`,
              border: "3px solid black",
            }}
            onClick={openSetColor}
          ></div>
          {openSetColorState && (
            <div className={style.setColor}>
              {colors.map((color) => (
                <div
                  style={{
                    backgroundColor: color,
                    width: "30px",
                    height: "30px",
                    borderRadius: "50%",
                    margin: "5px",
                  }}
                  data-color={color}
                  onClick={(event) => {
                    setColor(event);
                    openSetColor();
                  }}
                ></div>
              ))}
            </div>
          )}
        </div>
        <div className={style.tool4}>
          <div
            style={{
              width: "50px",
              height: "30px",
              backgroundColor: "white",
              border: "3px solid black",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
            onClick={openSetWidth}
          >
            <div
              style={{
                width: `${widthState}px`,
                height: `${widthState}px`,
                backgroundColor: "black",
                borderRadius: "50%",
              }}
            ></div>
          </div>
          {openSetWidthState && (
            <div className={style.setWidth}>
              <input
                type="range"
                min="1"
                max="20"
                defaultValue={widthState}
                step="0.1"
                onMouseUp={(event) => {
                  setWidth(event);
                  openSetWidth();
                }}
              />
            </div>
          )}
        </div>
        <div className={style.tool5}>
          <img
            src={process.env.PUBLIC_URL + `/icon/back.svg`}
            alt="back"
            onClick={goBack}
          />
        </div>
        <div className={style.tool6}>
          <img
            src={process.env.PUBLIC_URL + `/icon/clear.svg`}
            alt="clear"
            onClick={initCanvas}
          />
        </div>
      </div>
    </div>
  );
}