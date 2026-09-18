-- ============================================================================
--  Vista de entrenamiento del modelo predictivo de genero musical.
--  Base: Chinook (Supabase)
--
--  Campos requeridos por el enunciado:
--    tipo_correo  : tipo de correo electronico del cliente
--    pais         : pais de origen
--    ciudad       : ciudad de origen
--    genero       : genero musical  (variable objetivo)
--
--  Campo adicional usado como feature:
--    proveedor_correo : proveedor del correo (gmail, yahoo, apple, ...)
--
--  Grano: una fila por track comprado (invoice_line). Se eligio este grano
--  porque a nivel cliente el dataset degenera: 45 de 59 clientes tienen "Rock"
--  como genero dominante (76%), y el modelo colapsaria a predecir siempre Rock.
-- ============================================================================

create or replace view public.vw_cliente_genero as
with dom as (
  select
    c.customer_id,
    c.country,
    c.city,
    lower(trim(split_part(c.email, '@', 2))) as dominio
  from public.customer c
  where c.email is not null and c.email like '%@%'
),
clasificado as (
  select
    d.customer_id,
    d.country,
    d.city,
    d.dominio,
    case
      when split_part(d.dominio, '.', 1) = 'yachoo' then 'yahoo'  -- typo en el dataset
      else split_part(d.dominio, '.', 1)
    end as proveedor
  from dom d
)
select
  case
    when cl.dominio ~ '\.(gov|gob|mil)(\.|$)' then 'gubernamental'
    when cl.dominio ~ '\.(edu|ac)(\.|$)'      then 'educativo'
    when cl.proveedor in (
      'gmail','hotmail','yahoo','outlook','live','msn','aol','rediff','jubii',
      'wp','sapo','uol','terra','bol','ig','gmx','yandex','zoho','protonmail',
      'icloud','mail'
    ) then 'gratuito'
    when cl.proveedor in (
      'shaw','rogers','surfeu','comcast','telus','bell','videotron','sympatico',
      'verizon','att','cox','charter','orange','wanadoo','t-online','virgin',
      'sky','btinternet'
    ) then 'isp'
    else 'corporativo'
  end        as tipo_correo,
  cl.proveedor as proveedor_correo,
  cl.country   as pais,
  cl.city      as ciudad,
  g.name       as genero
from clasificado cl
join public.invoice      i  on i.customer_id = cl.customer_id
join public.invoice_line il on il.invoice_id = i.invoice_id
join public.track        t  on t.track_id    = il.track_id
join public.genre        g  on g.genre_id    = t.genre_id;

comment on view public.vw_cliente_genero is
  'Dataset de entrenamiento: perfil del cliente (tipo/proveedor de correo, pais, ciudad) vs genero musical comprado. Una fila por track comprado.';
