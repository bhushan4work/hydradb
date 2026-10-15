use std::collections::BTreeMap;

use crate::QueryFailureReason;
use crate::{GraphError, QueryColumn, QueryFloat, Result, VertexId, VertexPropertyValue};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecallCertColumn {
    Reached,
    Total,
    Recall,
    Pass,
}

impl RecallCertColumn {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Reached => "reached",
            Self::Total => "total",
            Self::Recall => "recall",
            Self::Pass => "pass",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RecallCertProjection {
    Yielded {
        column: RecallCertColumn,
        alias: Option<String>,
    },
    CountAll {
        alias: Option<String>,
    },
}

impl RecallCertProjection {
    pub(crate) fn column(&self) -> QueryColumn {
        match self {
            Self::Yielded { column, alias } => {
                QueryColumn::new(alias.as_deref().unwrap_or_else(|| column.name()))
            }
            Self::CountAll { alias } => QueryColumn::new(alias.as_deref().unwrap_or("count(*)")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecallCertProcedure {
    pub(crate) source_nodes: Vec<VertexId>,
    pub(crate) label_nodes: Vec<VertexId>,
    pub(crate) rel_type: String,
    pub(crate) max_len: u8,
    pub(crate) threshold: f64,
    pub(crate) projections: Vec<RecallCertProjection>,
}

impl RecallCertProcedure {
    pub(crate) fn is_count_all(&self) -> bool {
        matches!(
            self.projections.as_slice(),
            [RecallCertProjection::CountAll { .. }]
        )
    }

    pub(crate) fn columns(&self) -> Vec<QueryColumn> {
        self.projections
            .iter()
            .map(RecallCertProjection::column)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativePathProcedureKind {
    SinglePair,
    SingleSource,
    MultiSource,
}

impl NativePathProcedureKind {
    pub(crate) const fn procedure_name(self) -> &'static str {
        match self {
            Self::SinglePair => "algo.SPpaths",
            Self::SingleSource => "algo.SSpaths",
            Self::MultiSource => "algo.MSpaths",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativePathDirection {
    Incoming,
    Outgoing,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativePathProjection {
    Path,
    Weight,
    Cost,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativePathNodeSelector {
    pub(crate) label: String,
    pub(crate) property: String,
    pub(crate) values: Vec<String>,
}

impl NativePathProjection {
    pub(crate) fn column(self) -> QueryColumn {
        QueryColumn::new(match self {
            Self::Path => "path",
            Self::Weight => "pathWeight",
            Self::Cost => "pathCost",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NativePathProcedure {
    pub(crate) kind: NativePathProcedureKind,
    pub(crate) source: VertexId,
    pub(crate) target: Option<VertexId>,
    pub(crate) source_selector: Option<NativePathNodeSelector>,
    pub(crate) target_selector: Option<NativePathNodeSelector>,
    pub(crate) pairwise: bool,
    pub(crate) fair_relationship_variants: bool,
    pub(crate) rel_types: Vec<String>,
    pub(crate) direction: NativePathDirection,
    pub(crate) max_len: u8,
    pub(crate) weight_property: Option<String>,
    pub(crate) cost_property: Option<String>,
    pub(crate) max_cost: Option<QueryFloat>,
    pub(crate) path_count: u64,
    pub(crate) result_limit: Option<u64>,
    pub(crate) projections: Vec<NativePathProjection>,
}

struct ParsedNativePathCall {
    kind: NativePathProcedureKind,
    config: BTreeMap<String, ConfigValue>,
    projections: Vec<NativePathProjection>,
}

pub(crate) fn is_native_path_procedure(query: &str) -> bool {
    let query = query.trim_start();
    query
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("CALL"))
        && query[4..]
            .trim_start()
            .to_ascii_lowercase()
            .starts_with("algo.")
}

pub(crate) fn parse_native_path_procedure(
    query: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
    max_traversal_hops: u8,
) -> Result<Option<NativePathProcedure>> {
    let Some(parsed) = parse_native_path_call(query)? else {
        return Ok(None);
    };
    let ParsedNativePathCall {
        kind,
        config,
        projections,
    } = parsed;

    let (source, target, source_selector, target_selector, pairwise) = match kind {
        NativePathProcedureKind::SinglePair => {
            let source = required_vertex(&config, "sourceNode", parameters)?;
            let target = optional_vertex(&config, "targetNode", parameters)?;
            if target.is_none() {
                return path_parse_error("algo.SPpaths requires targetNode");
            }
            (source, target, None, None, false)
        }
        NativePathProcedureKind::SingleSource => {
            let source = required_vertex(&config, "sourceNode", parameters)?;
            if optional_vertex(&config, "targetNode", parameters)?.is_some() {
                return path_parse_error("algo.SSpaths does not accept targetNode");
            }
            (source, None, None, None, false)
        }
        NativePathProcedureKind::MultiSource => {
            let source_selector = required_selector(&config, "source")?;
            let target_selector = optional_selector(&config, "target", &source_selector)?;
            let pairwise = config_bool(&config, "pairwise")?.unwrap_or(false);
            if pairwise && target_selector.is_none() {
                return path_parse_error("algo.MSpaths pairwise mode requires targetValues");
            }
            (0, None, Some(source_selector), target_selector, pairwise)
        }
    };
    let rel_types =
        config_string_list(&config, "relTypes")?.ok_or_else(|| GraphError::UnsupportedQuery {
            reason: QueryFailureReason::Procedure,
            dialect: "OpenCypher",
            feature: "native path procedures require a non-empty relTypes list".to_string(),
        })?;
    if rel_types.is_empty() {
        return path_parse_error("native path procedures require a non-empty relTypes list");
    }
    for edge_type in &rel_types {
        crate::validate_component("edge_type", edge_type)?;
    }
    let direction_value =
        config_string(&config, "relDirection", parameters)?.map(|value| value.to_ascii_lowercase());
    let direction = match direction_value.as_deref() {
        None | Some("outgoing") => NativePathDirection::Outgoing,
        Some("incoming") => NativePathDirection::Incoming,
        Some("both") => NativePathDirection::Both,
        Some(_) => {
            return path_parse_error("relDirection must be 'incoming', 'outgoing', or 'both'")
        }
    };
    let max_len =
        config_u64(&config, "maxLen", parameters)?.unwrap_or(u64::from(max_traversal_hops));
    if max_len > u64::from(max_traversal_hops) {
        return Err(GraphError::AdmissionRejected {
            operation: "native_path_max_len",
            actual: max_len,
            limit: u64::from(max_traversal_hops),
        });
    }
    let path_count = config_u64(&config, "pathCount", parameters)?.unwrap_or(1);
    let result_limit = config_u64(&config, "resultLimit", parameters)?;
    if result_limit == Some(0) {
        return path_parse_error("resultLimit must be greater than zero");
    }
    let weight_property = config_string(&config, "weightProp", parameters)?;
    let cost_property = config_string(&config, "costProp", parameters)?;
    let max_cost = config_number(&config, "maxCost", parameters)?.map(QueryFloat);
    let fair_relationship_variants =
        config_bool(&config, "fairRelationshipVariants")?.unwrap_or(false);
    if fair_relationship_variants && (kind != NativePathProcedureKind::MultiSource || !pairwise) {
        return path_parse_error(
            "fairRelationshipVariants is only supported by pairwise algo.MSpaths",
        );
    }
    if fair_relationship_variants
        && (weight_property.is_some() || cost_property.is_some() || max_cost.is_some())
    {
        return path_parse_error(
            "fairRelationshipVariants requires an unweighted pairwise algo.MSpaths query",
        );
    }
    let known = [
        "sourceNode",
        "targetNode",
        "relTypes",
        "relDirection",
        "maxLen",
        "weightProp",
        "costProp",
        "maxCost",
        "pathCount",
        "resultLimit",
        "sourceLabel",
        "sourceProperty",
        "sourceValues",
        "targetLabel",
        "targetProperty",
        "targetValues",
        "pairwise",
        "fairRelationshipVariants",
    ];
    if let Some(unknown) = config.keys().find(|key| !known.contains(&key.as_str())) {
        return path_parse_error(format!("unknown native path option {unknown}"));
    }
    let multi_source_options = [
        "sourceLabel",
        "sourceProperty",
        "sourceValues",
        "targetLabel",
        "targetProperty",
        "targetValues",
        "pairwise",
        "fairRelationshipVariants",
    ];
    if kind != NativePathProcedureKind::MultiSource {
        if let Some(option) = multi_source_options
            .iter()
            .find(|option| config.contains_key(**option))
        {
            return path_parse_error(format!("{option} is only supported by algo.MSpaths"));
        }
    } else if let Some(option) = ["sourceNode", "targetNode"]
        .iter()
        .find(|option| config.contains_key(**option))
    {
        return path_parse_error(format!(
            "{option} is not supported by algo.MSpaths; use indexed selectors"
        ));
    }
    Ok(Some(NativePathProcedure {
        kind,
        source,
        target,
        source_selector,
        target_selector,
        pairwise,
        fair_relationship_variants,
        rel_types,
        direction,
        max_len: max_len as u8,
        weight_property,
        cost_property,
        max_cost,
        path_count,
        result_limit,
        projections,
    }))
}

#[cfg(feature = "client-api")]
pub(crate) fn parse_native_path_procedure_columns(query: &str) -> Result<Option<Vec<QueryColumn>>> {
    if let Some(parsed) = parse_native_path_call(query)? {
        return Ok(Some(
            parsed
                .projections
                .into_iter()
                .map(NativePathProjection::column)
                .collect(),
        ));
    }
    if let Some(parsed) = parse_recall_cert_call(query)? {
        return Ok(Some(
            parsed
                .projections
                .iter()
                .map(RecallCertProjection::column)
                .collect(),
        ));
    }
    Ok(None)
}

#[derive(Clone, Debug)]
struct ParsedRecallCertCall {
    source_nodes: ConfigValue,
    label_nodes: ConfigValue,
    rel_type: ConfigValue,
    max_len: ConfigValue,
    threshold: ConfigValue,
    projections: Vec<RecallCertProjection>,
}

pub(crate) fn parse_recall_cert_procedure(
    query: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
    list_parameters: &BTreeMap<String, Vec<VertexPropertyValue>>,
    max_traversal_hops: u8,
) -> Result<Option<RecallCertProcedure>> {
    let Some(parsed) = parse_recall_cert_call(query)? else {
        return Ok(None);
    };
    let source_nodes = resolve_node_list(
        &parsed.source_nodes,
        "sourceNodes",
        parameters,
        list_parameters,
    )?;
    let label_nodes = resolve_node_list(
        &parsed.label_nodes,
        "labelNodes",
        parameters,
        list_parameters,
    )?;
    let rel_type = resolve_rel_type(&parsed.rel_type, "relType", parameters)?;
    let max_len = resolve_max_len(&parsed.max_len, "maxLen", parameters, max_traversal_hops)?;
    let threshold = resolve_threshold(&parsed.threshold, "threshold", parameters)?;

    Ok(Some(RecallCertProcedure {
        source_nodes,
        label_nodes,
        rel_type,
        max_len,
        threshold,
        projections: parsed.projections,
    }))
}

fn parse_recall_cert_call(query: &str) -> Result<Option<ParsedRecallCertCall>> {
    if !is_native_path_procedure(query) {
        return Ok(None);
    }
    let mut parser = Parser::new(query)?;
    parser.keyword("CALL")?;
    parser.identifier_eq("algo")?;
    parser.token(Token::Dot)?;
    let procedure = parser.identifier()?;
    if !procedure.eq_ignore_ascii_case("RecallCert") {
        return Ok(None);
    }
    parser.token(Token::LeftParen)?;
    let (source_nodes, label_nodes, rel_type, max_len, threshold) = if parser
        .optional(Token::LeftBrace)
    {
        let mut config = BTreeMap::new();
        if !parser.optional(Token::RightBrace) {
            loop {
                let key = parser.identifier()?;
                parser.token(Token::Colon)?;
                let value = parser.value()?;
                if config.insert(key.clone(), value).is_some() {
                    return path_parse_error(format!("duplicate recall cert option {key}"));
                }
                if parser.optional(Token::RightBrace) {
                    break;
                }
                parser.token(Token::Comma)?;
            }
        }
        let source_nodes =
            config
                .remove("sourceNodes")
                .ok_or_else(|| GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: "sourceNodes".to_string(),
                })?;
        let label_nodes =
            config
                .remove("labelNodes")
                .ok_or_else(|| GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: "labelNodes".to_string(),
                })?;
        let rel_type =
            config
                .remove("relType")
                .ok_or_else(|| GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: "relType".to_string(),
                })?;
        let max_len = config
            .remove("maxLen")
            .ok_or_else(|| GraphError::MissingQueryParameter {
                dialect: "OpenCypher",
                name: "maxLen".to_string(),
            })?;
        let threshold =
            config
                .remove("threshold")
                .ok_or_else(|| GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: "threshold".to_string(),
                })?;
        if let Some(unknown) = config.keys().next() {
            return path_parse_error(format!("unknown recall cert option {unknown}"));
        }
        (source_nodes, label_nodes, rel_type, max_len, threshold)
    } else {
        let source_nodes = parser.value()?;
        parser.token(Token::Comma)?;
        let label_nodes = parser.value()?;
        parser.token(Token::Comma)?;
        let rel_type = parser.value()?;
        parser.token(Token::Comma)?;
        let max_len = parser.value()?;
        parser.token(Token::Comma)?;
        let threshold = parser.value()?;
        (source_nodes, label_nodes, rel_type, max_len, threshold)
    };
    parser.token(Token::RightParen)?;
    parser.keyword("YIELD")?;
    let yielded_columns = if parser.optional(Token::Star) {
        vec![
            RecallCertColumn::Reached,
            RecallCertColumn::Total,
            RecallCertColumn::Recall,
            RecallCertColumn::Pass,
        ]
    } else {
        let mut cols = Vec::new();
        loop {
            let name = parser.identifier()?;
            let col = match name.to_ascii_lowercase().as_str() {
                "reached" => RecallCertColumn::Reached,
                "total" => RecallCertColumn::Total,
                "recall" => RecallCertColumn::Recall,
                "pass" => RecallCertColumn::Pass,
                other => return path_parse_error(format!("unknown recall cert column {other}")),
            };
            if cols.contains(&col) {
                return path_parse_error(format!("duplicate yield column {name}"));
            }
            cols.push(col);
            if !parser.optional(Token::Comma) {
                break;
            }
        }
        cols
    };

    let projections = if parser.optional_keyword("RETURN") {
        if parser.optional(Token::Star) {
            yielded_columns
                .into_iter()
                .map(|col| RecallCertProjection::Yielded {
                    column: col,
                    alias: None,
                })
                .collect()
        } else {
            let mut projs = Vec::new();
            loop {
                if parser.peek().is_some_and(|t| match t {
                    Token::Identifier(s) => s.eq_ignore_ascii_case("count"),
                    _ => false,
                }) && parser.tokens.get(parser.offset + 1) == Some(&Token::LeftParen)
                    && parser.tokens.get(parser.offset + 2) == Some(&Token::Star)
                    && parser.tokens.get(parser.offset + 3) == Some(&Token::RightParen)
                {
                    parser.offset += 4;
                    let alias = if parser.optional_keyword("AS") {
                        Some(parser.identifier()?)
                    } else {
                        None
                    };
                    projs.push(RecallCertProjection::CountAll { alias });
                } else {
                    let name = parser.identifier()?;
                    let col = match name.to_ascii_lowercase().as_str() {
                        "reached" => RecallCertColumn::Reached,
                        "total" => RecallCertColumn::Total,
                        "recall" => RecallCertColumn::Recall,
                        "pass" => RecallCertColumn::Pass,
                        other => {
                            return path_parse_error(format!("unknown recall cert column {other}"))
                        }
                    };
                    if !yielded_columns.contains(&col) {
                        return path_parse_error("RETURN may only reference yielded columns");
                    }
                    let alias = if parser.optional_keyword("AS") {
                        Some(parser.identifier()?)
                    } else {
                        None
                    };
                    projs.push(RecallCertProjection::Yielded { column: col, alias });
                }
                if !parser.optional(Token::Comma) {
                    break;
                }
            }
            projs
        }
    } else {
        yielded_columns
            .into_iter()
            .map(|col| RecallCertProjection::Yielded {
                column: col,
                alias: None,
            })
            .collect()
    };

    parser.optional(Token::Semicolon);
    parser.end()?;

    Ok(Some(ParsedRecallCertCall {
        source_nodes,
        label_nodes,
        rel_type,
        max_len,
        threshold,
        projections,
    }))
}

fn resolve_node_list(
    value: &ConfigValue,
    name: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
    list_parameters: &BTreeMap<String, Vec<VertexPropertyValue>>,
) -> Result<Vec<VertexId>> {
    match value {
        ConfigValue::List(items) => {
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    ConfigValue::Integer(id) => {
                        result.push(non_negative_u64(*id, name)?);
                    }
                    ConfigValue::Parameter(param) => {
                        let prop = parameters.get(param).ok_or_else(|| {
                            GraphError::MissingQueryParameter {
                                dialect: "OpenCypher",
                                name: param.clone(),
                            }
                        })?;
                        match prop {
                            VertexPropertyValue::Integer(id) => {
                                result.push(*id);
                            }
                            VertexPropertyValue::SignedInteger(id) => {
                                result.push(non_negative_u64(*id, name)?);
                            }
                            _ => {
                                return path_parse_error(format!(
                                    "{name} must contain integer node ids"
                                ))
                            }
                        }
                    }
                    _ => return path_parse_error(format!("{name} must contain integer node ids")),
                }
            }
            Ok(result)
        }
        ConfigValue::Parameter(param) => {
            if let Some(list) = list_parameters.get(param) {
                let mut result = Vec::with_capacity(list.len());
                for prop in list {
                    match prop {
                        VertexPropertyValue::Integer(id) => {
                            result.push(*id);
                        }
                        VertexPropertyValue::SignedInteger(id) => {
                            result.push(non_negative_u64(*id, name)?);
                        }
                        _ => {
                            return path_parse_error(format!(
                                "{name} must contain integer node ids"
                            ))
                        }
                    }
                }
                return Ok(result);
            }
            if let Some(prop) = parameters.get(param) {
                match prop {
                    VertexPropertyValue::Integer(id) => return Ok(vec![*id]),
                    VertexPropertyValue::SignedInteger(id) => {
                        return Ok(vec![non_negative_u64(*id, name)?]);
                    }
                    _ => {
                        return path_parse_error(format!(
                            "{name} must be an integer or list of integer node ids"
                        ))
                    }
                }
            }
            Err(GraphError::MissingQueryParameter {
                dialect: "OpenCypher",
                name: param.clone(),
            })
        }
        ConfigValue::Identifier(ident) => {
            if let Some(list) = list_parameters.get(ident) {
                let mut result = Vec::with_capacity(list.len());
                for prop in list {
                    match prop {
                        VertexPropertyValue::Integer(id) => {
                            result.push(*id);
                        }
                        VertexPropertyValue::SignedInteger(id) => {
                            result.push(non_negative_u64(*id, name)?);
                        }
                        _ => {
                            return path_parse_error(format!(
                                "{name} must contain integer node ids"
                            ))
                        }
                    }
                }
                return Ok(result);
            }
            if let Some(prop) = parameters.get(ident) {
                match prop {
                    VertexPropertyValue::Integer(id) => return Ok(vec![*id]),
                    VertexPropertyValue::SignedInteger(id) => {
                        return Ok(vec![non_negative_u64(*id, name)?]);
                    }
                    _ => {
                        return path_parse_error(format!(
                            "{name} must be an integer or list of integer node ids"
                        ))
                    }
                }
            }
            path_parse_error(format!("{name} must be a list of integer node ids"))
        }
        ConfigValue::Integer(id) => Ok(vec![non_negative_u64(*id, name)?]),
        _ => path_parse_error(format!("{name} must be a list of integer node ids")),
    }
}

fn resolve_rel_type(
    value: &ConfigValue,
    name: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<String> {
    let rel_type = match value {
        ConfigValue::String(s) => s.clone(),
        ConfigValue::Identifier(s) => {
            if let Some(VertexPropertyValue::String(val)) = parameters.get(s) {
                val.clone()
            } else {
                s.clone()
            }
        }
        ConfigValue::Parameter(param) => match parameters.get(param) {
            Some(VertexPropertyValue::String(s)) => s.clone(),
            Some(_) => return path_parse_error(format!("{name} parameter must be a string")),
            None => {
                return Err(GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: param.clone(),
                })
            }
        },
        _ => return path_parse_error(format!("{name} must be a string or identifier")),
    };
    if rel_type.is_empty() {
        return path_parse_error(format!("{name} must be a non-empty string"));
    }
    crate::validate_component("edge_type", &rel_type)?;
    Ok(rel_type)
}

fn resolve_max_len(
    value: &ConfigValue,
    name: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
    max_traversal_hops: u8,
) -> Result<u8> {
    let max_len_u64 = match value {
        ConfigValue::Integer(i) => non_negative_u64(*i, name)?,
        ConfigValue::Parameter(param) => match parameters.get(param) {
            Some(VertexPropertyValue::Integer(i)) => *i,
            Some(VertexPropertyValue::SignedInteger(i)) => non_negative_u64(*i, name)?,
            Some(_) => return path_parse_error(format!("{name} parameter must be an integer")),
            None => {
                return Err(GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: param.clone(),
                })
            }
        },
        ConfigValue::Identifier(ident) => match parameters.get(ident) {
            Some(VertexPropertyValue::Integer(i)) => *i,
            Some(VertexPropertyValue::SignedInteger(i)) => non_negative_u64(*i, name)?,
            Some(_) => return path_parse_error(format!("{name} parameter must be an integer")),
            None => return path_parse_error(format!("{name} must be a non-negative integer")),
        },
        _ => return path_parse_error(format!("{name} must be a non-negative integer")),
    };
    if max_len_u64 > u64::from(max_traversal_hops) {
        return Err(GraphError::AdmissionRejected {
            operation: "native_path_max_len",
            actual: max_len_u64,
            limit: u64::from(max_traversal_hops),
        });
    }
    Ok(max_len_u64 as u8)
}

fn resolve_threshold(
    value: &ConfigValue,
    name: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<f64> {
    let t = match value {
        ConfigValue::Float(f) => *f,
        ConfigValue::Integer(i) => *i as f64,
        ConfigValue::Parameter(param) => match parameters.get(param) {
            Some(VertexPropertyValue::Float(f)) => f.0,
            Some(VertexPropertyValue::Integer(i)) => *i as f64,
            Some(VertexPropertyValue::SignedInteger(i)) => *i as f64,
            Some(_) => return path_parse_error(format!("{name} parameter must be numeric")),
            None => {
                return Err(GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: param.clone(),
                })
            }
        },
        ConfigValue::Identifier(ident) => match parameters.get(ident) {
            Some(VertexPropertyValue::Float(f)) => f.0,
            Some(VertexPropertyValue::Integer(i)) => *i as f64,
            Some(VertexPropertyValue::SignedInteger(i)) => *i as f64,
            Some(_) => return path_parse_error(format!("{name} parameter must be numeric")),
            None => return path_parse_error(format!("{name} must be numeric")),
        },
        _ => return path_parse_error(format!("{name} must be numeric")),
    };
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return path_parse_error(format!("{name} must be finite and between 0.0 and 1.0"));
    }
    Ok(t)
}

fn parse_native_path_call(query: &str) -> Result<Option<ParsedNativePathCall>> {
    if !is_native_path_procedure(query) {
        return Ok(None);
    }
    let mut parser = Parser::new(query)?;
    parser.keyword("CALL")?;
    parser.identifier_eq("algo")?;
    parser.token(Token::Dot)?;
    let procedure = parser.identifier()?;
    let kind = if procedure.eq_ignore_ascii_case("SPpaths") {
        NativePathProcedureKind::SinglePair
    } else if procedure.eq_ignore_ascii_case("SSpaths") {
        NativePathProcedureKind::SingleSource
    } else if procedure.eq_ignore_ascii_case("MSpaths") {
        NativePathProcedureKind::MultiSource
    } else {
        return Ok(None);
    };
    parser.token(Token::LeftParen)?;
    let config = parser.config()?;
    parser.token(Token::RightParen)?;
    parser.keyword("YIELD")?;
    let yielded = parser.projections()?;
    parser.keyword("RETURN")?;
    let projections = parser.projections()?;
    parser.optional(Token::Semicolon);
    parser.end()?;
    for projection in &projections {
        if !yielded.contains(projection) {
            return path_parse_error("RETURN may only reference yielded path columns");
        }
    }
    Ok(Some(ParsedNativePathCall {
        kind,
        config,
        projections,
    }))
}

fn required_selector(
    config: &BTreeMap<String, ConfigValue>,
    prefix: &str,
) -> Result<NativePathNodeSelector> {
    optional_selector_parts(config, prefix, None)?.ok_or_else(|| {
        GraphError::MissingQueryParameter {
            dialect: "OpenCypher",
            name: format!("{prefix}Values"),
        }
    })
}

fn optional_selector(
    config: &BTreeMap<String, ConfigValue>,
    prefix: &str,
    defaults: &NativePathNodeSelector,
) -> Result<Option<NativePathNodeSelector>> {
    optional_selector_parts(config, prefix, Some(defaults))
}

fn optional_selector_parts(
    config: &BTreeMap<String, ConfigValue>,
    prefix: &str,
    defaults: Option<&NativePathNodeSelector>,
) -> Result<Option<NativePathNodeSelector>> {
    let values_key = format!("{prefix}Values");
    let Some(values) = config_string_list(config, &values_key)? else {
        let label_key = format!("{prefix}Label");
        let property_key = format!("{prefix}Property");
        if config.contains_key(&label_key) || config.contains_key(&property_key) {
            return path_parse_error(format!(
                "{values_key} is required when {prefix}Label or {prefix}Property is set"
            ));
        }
        return Ok(None);
    };
    if values.is_empty() {
        return path_parse_error(format!("{values_key} must not be empty"));
    }
    if values.iter().any(|value| value.is_empty()) {
        return path_parse_error(format!("{values_key} must not contain empty strings"));
    }
    let label_key = format!("{prefix}Label");
    let property_key = format!("{prefix}Property");
    let label = config_literal_string(config, &label_key)?
        .or_else(|| defaults.map(|selector| selector.label.clone()))
        .ok_or_else(|| GraphError::MissingQueryParameter {
            dialect: "OpenCypher",
            name: label_key,
        })?;
    let property = config_literal_string(config, &property_key)?
        .or_else(|| defaults.map(|selector| selector.property.clone()))
        .ok_or_else(|| GraphError::MissingQueryParameter {
            dialect: "OpenCypher",
            name: property_key,
        })?;
    crate::validate_component("label", &label)?;
    crate::validate_component("property", &property)?;
    Ok(Some(NativePathNodeSelector {
        label,
        property,
        values,
    }))
}

fn config_literal_string(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
) -> Result<Option<String>> {
    match config.get(key) {
        None | Some(ConfigValue::Null) => Ok(None),
        Some(ConfigValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => path_parse_error(format!("{key} must be a string literal")),
    }
}

fn config_bool(config: &BTreeMap<String, ConfigValue>, key: &str) -> Result<Option<bool>> {
    match config.get(key) {
        None | Some(ConfigValue::Null) => Ok(None),
        Some(ConfigValue::Identifier(value)) if value.eq_ignore_ascii_case("true") => {
            Ok(Some(true))
        }
        Some(ConfigValue::Identifier(value)) if value.eq_ignore_ascii_case("false") => {
            Ok(Some(false))
        }
        Some(_) => path_parse_error(format!("{key} must be true or false")),
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ConfigValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Parameter(String),
    List(Vec<ConfigValue>),
    Identifier(String),
}

fn required_vertex(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<VertexId> {
    optional_vertex(config, key, parameters)?.ok_or_else(|| GraphError::MissingQueryParameter {
        dialect: "OpenCypher",
        name: key.to_string(),
    })
}

fn optional_vertex(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<Option<VertexId>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    let value = match value {
        ConfigValue::Null => return Ok(None),
        ConfigValue::Integer(value) => *value,
        ConfigValue::Parameter(parameter) => match parameters.get(parameter) {
            Some(VertexPropertyValue::Integer(value)) => {
                return Ok(Some(*value));
            }
            Some(VertexPropertyValue::SignedInteger(value)) => *value,
            Some(_) => {
                return path_parse_error(format!("{key} must resolve to an integer node id"))
            }
            None => {
                return Err(GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: parameter.clone(),
                })
            }
        },
        _ => return path_parse_error(format!("{key} must be an integer node id")),
    };
    u64::try_from(value)
        .map(Some)
        .map_err(|_| GraphError::UnsupportedQuery {
            reason: QueryFailureReason::Procedure,
            dialect: "OpenCypher",
            feature: format!("{key} must be a non-negative integer node id"),
        })
}

fn config_string(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<Option<String>> {
    match config.get(key) {
        None | Some(ConfigValue::Null) => Ok(None),
        Some(ConfigValue::String(value)) => Ok(Some(value.clone())),
        Some(ConfigValue::Parameter(parameter)) => match parameters.get(parameter) {
            Some(VertexPropertyValue::String(value)) => Ok(Some(value.clone())),
            Some(_) => path_parse_error(format!("{key} parameter must be a string")),
            None => Err(GraphError::MissingQueryParameter {
                dialect: "OpenCypher",
                name: parameter.clone(),
            }),
        },
        Some(_) => path_parse_error(format!("{key} must be a string")),
    }
}

fn config_string_list(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
) -> Result<Option<Vec<String>>> {
    match config.get(key) {
        None | Some(ConfigValue::Null) => Ok(None),
        Some(ConfigValue::List(values)) => values
            .iter()
            .map(|value| match value {
                ConfigValue::String(value) => Ok(value.clone()),
                _ => path_parse_error(format!("{key} must be a list of strings")),
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => path_parse_error(format!("{key} must be a list of strings")),
    }
}

fn config_u64(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<Option<u64>> {
    match config.get(key) {
        None | Some(ConfigValue::Null) => Ok(None),
        Some(ConfigValue::Integer(value)) => non_negative_u64(*value, key).map(Some),
        Some(ConfigValue::Parameter(parameter)) => match parameters.get(parameter) {
            Some(VertexPropertyValue::Integer(value)) => Ok(Some(*value)),
            Some(VertexPropertyValue::SignedInteger(value)) => {
                non_negative_u64(*value, key).map(Some)
            }
            Some(_) => path_parse_error(format!("{key} parameter must be an integer")),
            None => Err(GraphError::MissingQueryParameter {
                dialect: "OpenCypher",
                name: parameter.clone(),
            }),
        },
        Some(_) => path_parse_error(format!("{key} must be an integer")),
    }
}

fn non_negative_u64(value: i64, key: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| GraphError::UnsupportedQuery {
        reason: QueryFailureReason::Procedure,
        dialect: "OpenCypher",
        feature: format!("{key} must be a non-negative integer"),
    })
}

fn config_number(
    config: &BTreeMap<String, ConfigValue>,
    key: &str,
    parameters: &BTreeMap<String, VertexPropertyValue>,
) -> Result<Option<f64>> {
    let value = match config.get(key) {
        None | Some(ConfigValue::Null) => return Ok(None),
        Some(ConfigValue::Integer(value)) => *value as f64,
        Some(ConfigValue::Float(value)) => *value,
        Some(ConfigValue::Parameter(parameter)) => match parameters.get(parameter) {
            Some(VertexPropertyValue::Integer(value)) => *value as f64,
            Some(VertexPropertyValue::SignedInteger(value)) => *value as f64,
            Some(VertexPropertyValue::Float(value)) => value.0,
            Some(_) => return path_parse_error(format!("{key} parameter must be numeric")),
            None => {
                return Err(GraphError::MissingQueryParameter {
                    dialect: "OpenCypher",
                    name: parameter.clone(),
                })
            }
        },
        Some(_) => return path_parse_error(format!("{key} must be numeric")),
    };
    if !value.is_finite() || value < 0.0 {
        return path_parse_error(format!("{key} must be finite and non-negative"));
    }
    Ok(Some(value))
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Identifier(String),
    Parameter(String),
    String(String),
    Integer(i64),
    Float(f64),
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Colon,
    Comma,
    Dot,
    Semicolon,
    Star,
}

struct Parser {
    tokens: Vec<Token>,
    offset: usize,
}

impl Parser {
    fn new(query: &str) -> Result<Self> {
        Ok(Self {
            tokens: lex(query)?,
            offset: 0,
        })
    }

    fn config(&mut self) -> Result<BTreeMap<String, ConfigValue>> {
        self.token(Token::LeftBrace)?;
        let mut values = BTreeMap::new();
        if self.optional(Token::RightBrace) {
            return Ok(values);
        }
        loop {
            let key = self.identifier()?;
            self.token(Token::Colon)?;
            let value = self.value()?;
            if values.insert(key.clone(), value).is_some() {
                return path_parse_error(format!("duplicate native path option {key}"));
            }
            if self.optional(Token::RightBrace) {
                return Ok(values);
            }
            self.token(Token::Comma)?;
        }
    }

    fn value(&mut self) -> Result<ConfigValue> {
        match self.next()? {
            Token::Identifier(value) if value.eq_ignore_ascii_case("null") => Ok(ConfigValue::Null),
            Token::Identifier(value) => Ok(ConfigValue::Identifier(value)),
            Token::Parameter(value) => Ok(ConfigValue::Parameter(value)),
            Token::String(value) => Ok(ConfigValue::String(value)),
            Token::Integer(value) => Ok(ConfigValue::Integer(value)),
            Token::Float(value) => Ok(ConfigValue::Float(value)),
            Token::LeftBracket => {
                let mut values = Vec::new();
                if self.optional(Token::RightBracket) {
                    return Ok(ConfigValue::List(values));
                }
                loop {
                    values.push(self.value()?);
                    if self.optional(Token::RightBracket) {
                        return Ok(ConfigValue::List(values));
                    }
                    self.token(Token::Comma)?;
                    if self.optional(Token::RightBracket) {
                        return Ok(ConfigValue::List(values));
                    }
                }
            }
            token => path_parse_error(format!("unexpected native path value {token:?}")),
        }
    }

    fn projections(&mut self) -> Result<Vec<NativePathProjection>> {
        let mut projections = Vec::new();
        loop {
            let projection = match self.identifier()?.to_ascii_lowercase().as_str() {
                "path" => NativePathProjection::Path,
                "pathweight" => NativePathProjection::Weight,
                "pathcost" => NativePathProjection::Cost,
                other => return path_parse_error(format!("unknown path projection {other}")),
            };
            if projections.contains(&projection) {
                return path_parse_error("duplicate path projection");
            }
            projections.push(projection);
            if !self.optional(Token::Comma) {
                return Ok(projections);
            }
        }
    }

    fn keyword(&mut self, expected: &str) -> Result<()> {
        let actual = self.identifier()?;
        if actual.eq_ignore_ascii_case(expected) {
            Ok(())
        } else {
            path_parse_error(format!("expected {expected}, found {actual}"))
        }
    }

    fn identifier_eq(&mut self, expected: &str) -> Result<()> {
        self.keyword(expected)
    }

    fn identifier(&mut self) -> Result<String> {
        match self.next()? {
            Token::Identifier(value) => Ok(value),
            token => path_parse_error(format!("expected identifier, found {token:?}")),
        }
    }

    fn token(&mut self, expected: Token) -> Result<()> {
        let actual = self.next()?;
        if actual == expected {
            Ok(())
        } else {
            path_parse_error(format!("expected {expected:?}, found {actual:?}"))
        }
    }

    fn optional(&mut self, expected: Token) -> bool {
        if self.tokens.get(self.offset) == Some(&expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn optional_keyword(&mut self, expected: &str) -> bool {
        if let Some(Token::Identifier(value)) = self.tokens.get(self.offset) {
            if value.eq_ignore_ascii_case(expected) {
                self.offset += 1;
                return true;
            }
        }
        false
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.offset)
    }

    fn next(&mut self) -> Result<Token> {
        let token =
            self.tokens
                .get(self.offset)
                .cloned()
                .ok_or_else(|| GraphError::QueryParse {
                    dialect: "OpenCypher",
                    reason: "unexpected end of query".to_string(),
                })?;
        self.offset += 1;
        Ok(token)
    }

    fn end(&self) -> Result<()> {
        if self.offset == self.tokens.len() {
            Ok(())
        } else {
            path_parse_error("unexpected tokens after native path procedure")
        }
    }
}

fn lex(query: &str) -> Result<Vec<Token>> {
    let chars = query.char_indices().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < chars.len() {
        let (byte, ch) = chars[cursor];
        if ch.is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        let punctuation = match ch {
            '(' => Some(Token::LeftParen),
            ')' => Some(Token::RightParen),
            '{' => Some(Token::LeftBrace),
            '}' => Some(Token::RightBrace),
            '[' => Some(Token::LeftBracket),
            ']' => Some(Token::RightBracket),
            ':' => Some(Token::Colon),
            ',' => Some(Token::Comma),
            '.' => Some(Token::Dot),
            ';' => Some(Token::Semicolon),
            '*' => Some(Token::Star),
            _ => None,
        };
        if let Some(token) = punctuation {
            tokens.push(token);
            cursor += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            let quote = ch;
            let mut value = String::new();
            cursor += 1;
            let mut closed = false;
            while cursor < chars.len() {
                let (_, current) = chars[cursor];
                cursor += 1;
                if current == quote {
                    closed = true;
                    break;
                }
                if current == '\\' {
                    let Some((_, escaped)) = chars.get(cursor).copied() else {
                        return path_parse_error("unterminated string escape");
                    };
                    cursor += 1;
                    value.push(escaped);
                } else {
                    value.push(current);
                }
            }
            if !closed {
                return path_parse_error("unterminated string literal");
            }
            tokens.push(Token::String(value));
            continue;
        }
        if ch == '$' {
            let start = byte + 1;
            cursor += 1;
            let end = consume_identifier(query, &chars, &mut cursor, start)?;
            tokens.push(Token::Parameter(query[start..end].to_string()));
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = byte;
            cursor += 1;
            let end = consume_identifier(query, &chars, &mut cursor, start)?;
            tokens.push(Token::Identifier(query[start..end].to_string()));
            continue;
        }
        if ch.is_ascii_digit() || ch == '-' {
            let start = byte;
            cursor += 1;
            let mut saw_dot = false;
            while cursor < chars.len() {
                let (_, current) = chars[cursor];
                if current == '.' && !saw_dot {
                    saw_dot = true;
                    cursor += 1;
                } else if current.is_ascii_digit() {
                    cursor += 1;
                } else {
                    break;
                }
            }
            let end = chars.get(cursor).map_or(query.len(), |(byte, _)| *byte);
            let raw = &query[start..end];
            if saw_dot {
                let value = raw.parse::<f64>().map_err(|_| GraphError::QueryParse {
                    dialect: "OpenCypher",
                    reason: format!("invalid number {raw}"),
                })?;
                tokens.push(Token::Float(value));
            } else {
                let value = raw.parse::<i64>().map_err(|_| GraphError::QueryParse {
                    dialect: "OpenCypher",
                    reason: format!("invalid integer {raw}"),
                })?;
                tokens.push(Token::Integer(value));
            }
            continue;
        }
        return path_parse_error(format!("unexpected character {ch:?}"));
    }
    Ok(tokens)
}

fn consume_identifier(
    query: &str,
    chars: &[(usize, char)],
    cursor: &mut usize,
    start: usize,
) -> Result<usize> {
    while *cursor < chars.len()
        && (chars[*cursor].1.is_ascii_alphanumeric() || chars[*cursor].1 == '_')
    {
        *cursor += 1;
    }
    let end = chars.get(*cursor).map_or(query.len(), |(byte, _)| *byte);
    if end == start {
        return path_parse_error("expected identifier");
    }
    Ok(end)
}

fn path_parse_error<T>(reason: impl Into<String>) -> Result<T> {
    Err(GraphError::QueryParse {
        dialect: "OpenCypher",
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_pair_native_path_call() {
        let parameters = BTreeMap::from([
            ("source".to_string(), VertexPropertyValue::Integer(7)),
            ("target".to_string(), VertexPropertyValue::Integer(11)),
            ("count".to_string(), VertexPropertyValue::Integer(2)),
        ]);
        let parsed = parse_native_path_procedure(
            "CALL algo.SPpaths({sourceNode: $source, targetNode: $target, relTypes: ['RELATES'], maxLen: 3, relDirection: 'both', pathCount: $count}) YIELD path, pathWeight, pathCost RETURN path, pathWeight, pathCost",
            &parameters,
            10,
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.kind, NativePathProcedureKind::SinglePair);
        assert_eq!(parsed.source, 7);
        assert_eq!(parsed.target, Some(11));
        assert!(parsed.source_selector.is_none());
        assert!(parsed.target_selector.is_none());
        assert_eq!(parsed.rel_types, vec!["RELATES"]);
        assert_eq!(parsed.direction, NativePathDirection::Both);
        assert_eq!(parsed.max_len, 3);
        assert_eq!(parsed.path_count, 2);
    }

    #[test]
    fn parses_multi_source_native_path_call() {
        let parsed = parse_native_path_procedure(
            "CALL algo.MSpaths({sourceLabel: 'Entity', sourceProperty: 'name', sourceValues: ['alpha', 'beta'], targetValues: ['alpha', 'beta'], pairwise: true, relTypes: ['RELATES'], maxLen: 3, relDirection: 'both', pathCount: 5, fairRelationshipVariants: true, resultLimit: 100}) YIELD path RETURN path",
            &BTreeMap::new(),
            10,
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.kind, NativePathProcedureKind::MultiSource);
        assert!(parsed.pairwise);
        assert!(parsed.fair_relationship_variants);
        assert_eq!(parsed.path_count, 5);
        assert_eq!(parsed.result_limit, Some(100));
        assert_eq!(
            parsed.source_selector,
            Some(NativePathNodeSelector {
                label: "Entity".to_string(),
                property: "name".to_string(),
                values: vec!["alpha".to_string(), "beta".to_string()],
            })
        );
        assert_eq!(
            parsed.target_selector,
            Some(NativePathNodeSelector {
                label: "Entity".to_string(),
                property: "name".to_string(),
                values: vec!["alpha".to_string(), "beta".to_string()],
            })
        );
    }

    #[test]
    fn rejects_multi_source_options_on_scalar_procedures() {
        let error = parse_native_path_procedure(
            "CALL algo.SSpaths({sourceNode: 7, sourceValues: ['alpha'], relTypes: ['RELATES']}) YIELD path RETURN path",
            &BTreeMap::new(),
            10,
        )
        .unwrap_err();
        assert!(error.to_string().contains("only supported by algo.MSpaths"));
    }

    #[test]
    fn rejects_partial_multi_source_target_selector() {
        let error = parse_native_path_procedure(
            "CALL algo.MSpaths({sourceLabel: 'Entity', sourceProperty: 'name', sourceValues: ['alpha'], targetLabel: 'Entity', relTypes: ['RELATES']}) YIELD path RETURN path",
            &BTreeMap::new(),
            10,
        )
        .unwrap_err();
        assert!(error.to_string().contains("targetValues is required"));
    }

    #[test]
    fn rejects_path_depth_above_runtime_limit() {
        let error = parse_native_path_procedure(
            "CALL algo.SSpaths({sourceNode: 7, relTypes: ['RELATES'], maxLen: 11}) YIELD path RETURN path",
            &BTreeMap::new(),
            10,
        )
        .unwrap_err();
        assert!(matches!(error, GraphError::AdmissionRejected { .. }));
    }

    #[test]
    fn parses_recall_cert_positional_call_and_columns() {
        let query = "CALL algo.RecallCert([1, 2], [3, 4, 5], 'FOLLOWS', 3, 0.8) YIELD reached, total, recall, pass";
        let parsed = parse_recall_cert_procedure(query, &BTreeMap::new(), &BTreeMap::new(), 10)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.source_nodes, vec![1, 2]);
        assert_eq!(parsed.label_nodes, vec![3, 4, 5]);
        assert_eq!(parsed.rel_type, "FOLLOWS");
        assert_eq!(parsed.max_len, 3);
        assert!((parsed.threshold - 0.8).abs() < f64::EPSILON);
        assert!(!parsed.is_count_all());
        let col_names: Vec<String> = parsed.columns().into_iter().map(|c| c.name).collect();
        assert_eq!(col_names, vec!["reached", "total", "recall", "pass"]);
    }

    #[test]
    fn parses_recall_cert_map_call() {
        let query = "CALL algo.RecallCert({sourceNodes: [10], labelNodes: [20, 30], relType: 'KNOWS', maxLen: 2, threshold: 0.5}) YIELD reached, pass";
        let parsed = parse_recall_cert_procedure(query, &BTreeMap::new(), &BTreeMap::new(), 10)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.source_nodes, vec![10]);
        assert_eq!(parsed.label_nodes, vec![20, 30]);
        assert_eq!(parsed.rel_type, "KNOWS");
        assert_eq!(parsed.max_len, 2);
        assert!((parsed.threshold - 0.5).abs() < f64::EPSILON);
        let col_names: Vec<String> = parsed.columns().into_iter().map(|c| c.name).collect();
        assert_eq!(col_names, vec!["reached", "pass"]);
    }

    #[test]
    fn parses_recall_cert_yield_star_and_aliases() {
        let query = "CALL algo.RecallCert([1], [2], 'FOLLOWS', 1, 0.5) YIELD * RETURN recall AS metric, pass";
        let parsed = parse_recall_cert_procedure(query, &BTreeMap::new(), &BTreeMap::new(), 10)
            .unwrap()
            .unwrap();
        let col_names: Vec<String> = parsed.columns().into_iter().map(|c| c.name).collect();
        assert_eq!(col_names, vec!["metric", "pass"]);
    }

    #[test]
    fn parses_recall_cert_count_star() {
        let query1 =
            "CALL algo.RecallCert([1], [2], 'FOLLOWS', 1, 0.5) YIELD reached RETURN count(*)";
        let parsed1 = parse_recall_cert_procedure(query1, &BTreeMap::new(), &BTreeMap::new(), 10)
            .unwrap()
            .unwrap();
        assert!(parsed1.is_count_all());
        let col_names1: Vec<String> = parsed1.columns().into_iter().map(|c| c.name).collect();
        assert_eq!(col_names1, vec!["count(*)"]);

        let query2 = "CALL algo.RecallCert([1], [2], 'FOLLOWS', 1, 0.5) YIELD reached, total RETURN count(*) AS num_rows";
        let parsed2 = parse_recall_cert_procedure(query2, &BTreeMap::new(), &BTreeMap::new(), 10)
            .unwrap()
            .unwrap();
        assert!(parsed2.is_count_all());
        let col_names2: Vec<String> = parsed2.columns().into_iter().map(|c| c.name).collect();
        assert_eq!(col_names2, vec!["num_rows"]);
    }

    #[test]
    fn parses_recall_cert_parameters_and_list_parameters() {
        let mut params = BTreeMap::new();
        params.insert(
            "rel".to_string(),
            VertexPropertyValue::String("CONNECTS".to_string()),
        );
        params.insert("len".to_string(), VertexPropertyValue::Integer(4));
        params.insert(
            "thresh".to_string(),
            VertexPropertyValue::Float(crate::QueryFloat(0.75)),
        );
        params.insert("single_src".to_string(), VertexPropertyValue::Integer(42));

        let mut list_params = BTreeMap::new();
        list_params.insert(
            "label_list".to_string(),
            vec![
                VertexPropertyValue::Integer(100),
                VertexPropertyValue::Integer(101),
            ],
        );

        let query = "CALL algo.RecallCert($single_src, $label_list, $rel, $len, $thresh) YIELD reached, total, recall, pass";
        let parsed = parse_recall_cert_procedure(query, &params, &list_params, 10)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.source_nodes, vec![42]);
        assert_eq!(parsed.label_nodes, vec![100, 101]);
        assert_eq!(parsed.rel_type, "CONNECTS");
        assert_eq!(parsed.max_len, 4);
        assert!((parsed.threshold - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_recall_cert_invalid_threshold() {
        let query_neg = "CALL algo.RecallCert([1], [2], 'R', 1, -0.1) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_neg, &BTreeMap::new(), &BTreeMap::new(), 10).is_err()
        );

        let query_over = "CALL algo.RecallCert([1], [2], 'R', 1, 1.1) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_over, &BTreeMap::new(), &BTreeMap::new(), 10)
                .is_err()
        );
    }

    #[test]
    fn rejects_recall_cert_max_len_exceeding_hop_budget() {
        let query = "CALL algo.RecallCert([1], [2], 'R', 11, 0.5) YIELD reached";
        let error =
            parse_recall_cert_procedure(query, &BTreeMap::new(), &BTreeMap::new(), 10).unwrap_err();
        assert!(matches!(error, GraphError::AdmissionRejected { .. }));
    }

    #[test]
    fn rejects_recall_cert_negative_or_invalid_node_ids() {
        let query_neg = "CALL algo.RecallCert([-1], [2], 'R', 1, 0.5) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_neg, &BTreeMap::new(), &BTreeMap::new(), 10).is_err()
        );

        let query_str = "CALL algo.RecallCert(['abc'], [2], 'R', 1, 0.5) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_str, &BTreeMap::new(), &BTreeMap::new(), 10).is_err()
        );
    }

    #[test]
    fn rejects_recall_cert_empty_or_invalid_rel_type() {
        let query_empty = "CALL algo.RecallCert([1], [2], '', 1, 0.5) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_empty, &BTreeMap::new(), &BTreeMap::new(), 10)
                .is_err()
        );

        let query_invalid = "CALL algo.RecallCert([1], [2], 'bad/rel', 1, 0.5) YIELD reached";
        assert!(
            parse_recall_cert_procedure(query_invalid, &BTreeMap::new(), &BTreeMap::new(), 10)
                .is_err()
        );
    }

    #[test]
    fn rejects_recall_cert_missing_or_unknown_columns() {
        let query_unknown = "CALL algo.RecallCert([1], [2], 'R', 1, 0.5) YIELD unknown";
        assert!(
            parse_recall_cert_procedure(query_unknown, &BTreeMap::new(), &BTreeMap::new(), 10)
                .is_err()
        );

        let query_non_yielded =
            "CALL algo.RecallCert([1], [2], 'R', 1, 0.5) YIELD reached RETURN total";
        assert!(parse_recall_cert_procedure(
            query_non_yielded,
            &BTreeMap::new(),
            &BTreeMap::new(),
            10
        )
        .is_err());
    }
}
